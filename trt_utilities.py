#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from torch.cuda import nvtx
from collections import OrderedDict
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy import util
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
import tensorrt as trt
from logging import error, warning
from tqdm import tqdm
import copy

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}

class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, engine_path=None):
        # del self.engine
        del self.context
        del self.buffers
        del self.tensors
        # self.engine_path = engine_path

        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        """
        Builds a TensorRT engine for one or more STATIC profiles.
        """
        if not input_profile:
            error("Engine build failed: No input profile provided.")
            return 1

        try:
            network = network_from_onnx_path(
                onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            )
            if update_output_names:
                network = ModifyNetworkOutputs(network, update_output_names)

            builder = network[0]
            config = builder.create_builder_config()
            config.progress_monitor = TQDMProgressMonitor()
            
            # Static profile logic
            for p in input_profile:
                profile = Profile()
                for name, dims in p.items():
                    if not isinstance(dims, (list, tuple)) or len(dims) != 4:
                        error(f"Static profile shape must be a 4-element list/tuple (N,C,H,W). Got: {dims}")
                        return 1
                    # Set min, opt, and max to the *same* static shape
                    profile.add(name, min=dims, opt=dims, max=dims)
                
                calib_profile = profile.fill_defaults(network[1]).to_trt(builder, network[1])
                config.add_optimization_profile(calib_profile)

            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if enable_refit:
                config.set_flag(trt.BuilderFlag.REFIT)
            
            # WSL2/Docker performance fix: limit tactic sources
            if not enable_all_tactics:
                # Cast to int to fix the '|' operand error
                config.set_tactic_sources(int(trt.TacticSource.CUBLAS) | int(trt.TacticSource.CUDNN))

            engine = engine_from_network(network, config)
            save_engine(engine, path=self.engine_path)
            
            return 0 # Return 0 for success

        except Exception as e:
            error(f"Failed to build engine: {e}")
            return 1 # Return 1 for failure

    def load(self):
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        nvtx.range_push("allocate_buffers")
        
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]["shape"]
            else:
                shape = self.context.get_tensor_shape(name)

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        
        # Performance fix: Bind tensors once during allocation
        nvtx.range_push("bind_tensors")
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            # Performance fix: Asynchronous copy
            self.tensors[name].copy_(buf, non_blocking=True)

        # Performance fix: Removed the set_tensor_address loop
        nvtx.range_pop()
        
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        nvtx.range_pop()
        return self.tensors

    def __str__(self):
        out = ""
            
        # When raising errors in the upscaler, this str() called by comfy's execution.py,
        # but the engine won't have the attributes required for stringification
        # If str() also raises an error, comfy gets soft-locked, not running prompts until restarted
        if not hasattr(self.engine, "num_optimization_profiles") or not hasattr(self.engine, "num_bindings"):
            return out
        
        for opt_profile in range(self.engine.num_optimization_profiles):
            for binding_idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(binding_idx)
                shape = self.engine.get_profile_shape(opt_profile, name)
                out += f"\t{name} = {shape}\n"
        return out