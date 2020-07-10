/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2016-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

using namespace arm_compute;

const std::map<std::string, std::string> CLKernelLibraryEx::_kernel_program_map = {
    // ARMComputeEx kernels
    {"binary_logical_op", "binary_logical_op.cl"},
    {"embedding_lookup", "embedding_lookup.cl"},
    {"gather_ex", "gather_ex.cl"},
    {"gather_ex_1d", "gather_ex.cl"},
    {"gather_ex_1d_out", "gather_ex.cl"},
    {"gemmlowp_mm_midgard_ex", "gemmlowp_ex.cl"},
    {"hashtable_lookup", "hashtable_lookup.cl"},
    {"instance_normalization_ex", "instance_normalization_ex.cl"},
    {"multiply_scale_factor", "multiply_scale_factor.cl"},
    {"neg_tensor", "neg_tensor.cl"},
    {"quantization_symm8", "quantization_symm8.cl"},
    {"reduce_min_max", "reduce_operation.cl"},
    {"reduce_sum_mean", "reduce_operation.cl"},
    {"topkv2_init", "topkv2.cl"},
    {"topkv2_find_first_negative", "topkv2.cl"},
    {"topkv2_reorder_negatives", "topkv2.cl"},
    {"topkv2_store", "topkv2.cl"},
    {"radixsort_histogram", "topkv2_radixsort.cl"},
    {"radixsort_scanhistograms", "topkv2_radixsort.cl"},
    {"radixsort_pastehistograms", "topkv2_radixsort.cl"},
    {"radixsort_reorder", "topkv2_radixsort.cl"},
    {"topkv2_quicksort", "topkv2_quicksort.cl"},
    {"scale_factor_symm8", "scale_factor.cl"},
};

const std::map<std::string, std::string> CLKernelLibraryEx::_program_source_map = {
#ifdef EMBEDDED_KERNELS
    {
        "embedding_lookup.cl",
#include "./cl_kernels/embedding_lookup.clembed"
    },
    {
        "gather_ex.cl",
#include "./cl_kernels/gather_ex.clembed"
    },
    {
        "gemmlowp_ex.cl",
#include "./cl_kernels/gemmlowp_ex.clembed"
    },
    {
        "hashtable_lookup.cl",
#include "./cl_kernels/hashtable_lookup.clembed"
    },
    {
        "helpers.h",
#include "./cl_kernels/helpers.hembed"
    },
    {
        "helpers_asymm.h",
#include "./cl_kernels/helpers_asymm.hembed"
    },
    {
        "instance_normalization_ex.cl",
#include "./cl_kernels/instance_normalization_ex.clembed"
    },
    {
        "binary_logical_op.cl",
#include "./cl_kernels/binary_logical_op.clembed"
    },
    {
        "multiply_scale_factor.cl",
#include "./cl_kernels/multiply_scale_factor.clembed"
    },
    {
        "neg_tensor.cl",
#include "./cl_kernels/neg_tensor.clembed"
    },
    {
        "quantization_symm8.cl",
#include "./cl_kernels/quantization_symm8.clembed"
    },
    {
        "reduce_operation.cl",
#include "./cl_kernels/reduce_operation.clembed"
    },
    {
        "scale_factor.cl",
#include "./cl_kernels/scale_factor.clembed"
    },
    {
        "topkv2.cl",
#include "./cl_kernels/topkv2.clembed"
    },
    {
        "topkv2_radixsort.cl",
#include "./cl_kernels/topkv2_radixsort.clembed"
    },
    {
        "topkv2_quicksort.cl",
#include "./cl_kernels/topkv2_quicksort.clembed"
    },

#endif /* EMBEDDED_KERNELS */
};

CLKernelLibraryEx::CLKernelLibraryEx()
    : _context(), _device(), _kernel_path("."), _programs_map(), _built_programs_map()
{
  opencl_is_available(); // Make sure the OpenCL symbols are initialised *before* the
                         // CLKernelLibraryEx is built
}

CLKernelLibraryEx &CLKernelLibraryEx::get()
{
  static CLKernelLibraryEx _kernel_library;
  return _kernel_library;
}

Kernel CLKernelLibraryEx::create_kernel(const std::string &kernel_name,
                                        const StringSet &build_options_set) const
{
  // Find which program contains the kernel
  auto kernel_program_it = _kernel_program_map.find(kernel_name);

  if (_kernel_program_map.end() == kernel_program_it)
  {
    ARM_COMPUTE_ERROR_VAR("Kernel %s not found in the CLKernelLibrary", kernel_name.c_str());
  }
  std::string concat_str;

  if (fp16_supported())
  {
    concat_str += " -DARM_COMPUTE_OPENCL_FP16_ENABLED=1 ";
  }

  if (get_cl_version(_device) == CLVersion::CL20)
  {
    concat_str += " -cl-std=CL2.0 ";
  }
  else if (arm_non_uniform_workgroup_supported(_device))
  {
    concat_str += " -cl-arm-non-uniform-work-group-size ";
  }
  else
  {
    ARM_COMPUTE_ERROR("Non uniform workgroup size is not supported!!");
  }

  // Check if the program has been built before with same build options.
  const std::string program_name = kernel_program_it->second;
  const std::string build_options = stringify_set(build_options_set) + concat_str;

  const std::string built_program_name = program_name + "_" + build_options;
  auto built_program_it = _built_programs_map.find(built_program_name);

  cl::Program cl_program;

  if (_built_programs_map.end() != built_program_it)
  {
    // If program has been built, retrieve to create kernel from it
    cl_program = built_program_it->second;
  }
  else
  {
    // Get program
    Program program = load_program(program_name);

    // Build program
    cl_program = program.build(build_options);

    // Add built program to internal map
    _built_programs_map.emplace(built_program_name, cl_program);
  }

  // Create and return kernel
  return Kernel(kernel_name, cl_program);
}

void CLKernelLibraryEx::add_built_program(const std::string &built_program_name,
                                          cl::Program program)
{
  _built_programs_map.emplace(built_program_name, program);
}

bool CLKernelLibraryEx::fp16_supported() const { return ::fp16_supported(_device); }

bool CLKernelLibraryEx::int64_base_atomics_supported() const
{
  return device_supports_extension(_device, "cl_khr_int64_base_atomics");
}

const Program &CLKernelLibraryEx::load_program(const std::string &program_name) const
{
  const auto program_it = _programs_map.find(program_name);

  if (program_it != _programs_map.end())
  {
    return program_it->second;
  }

  Program program;

#ifdef EMBEDDED_KERNELS
  const auto program_source_it = _program_source_map.find(program_name);

  if (_program_source_map.end() == program_source_it)
  {
    ARM_COMPUTE_ERROR_VAR("Embedded program for %s does not exist.", program_name.c_str());
  }

  program = Program(_context, program_name, program_source_it->second);
#else  /* EMBEDDED_KERNELS */
  // Check for binary
  std::string source_name = _kernel_path + program_name;
  std::string binary_name = source_name + "bin";

  if (std::ifstream(binary_name).is_open())
  {
    const std::string program_binary = read_file(binary_name, true);
    program = Program(_context, _device, program_name,
                      std::vector<unsigned char>(program_binary.begin(), program_binary.end()));
  }
  else if (std::ifstream(source_name).is_open())
  {
    program = Program(_context, program_name, read_file(source_name, false));
  }
  else
  {
    ARM_COMPUTE_ERROR_VAR("Kernel file %s does not exist.", source_name.c_str());
  }
#endif /* EMBEDDED_KERNELS */

  // Insert program to program map
  const auto new_program = _programs_map.emplace(program_name, std::move(program));

  return new_program.first->second;
}

std::string CLKernelLibraryEx::stringify_set(const StringSet &s) const
{
  std::string concat_set;

#ifndef EMBEDDED_KERNELS
  concat_set += "-I" + _kernel_path + " ";
#endif /* EMBEDDED_KERNELS */

  // Concatenate set
  for (const auto &el : s)
  {
    concat_set += " " + el;
  }

  return concat_set;
}

std::string CLKernelLibraryEx::get_program_source(const std::string &program_name)
{
  const auto program_source_it = _program_source_map.find(program_name);

  if (program_source_it == _program_source_map.end())
  {
    ARM_COMPUTE_ERROR_VAR("Embedded program for %s does not exist.", program_name.c_str());
  }

  return program_source_it->second;
}

size_t CLKernelLibraryEx::max_local_workgroup_size(const cl::Kernel &kernel) const
{
  size_t result;

  size_t err = kernel.getWorkGroupInfo(_device, CL_KERNEL_WORK_GROUP_SIZE, &result);
  ARM_COMPUTE_ERROR_ON_MSG(
      err != 0,
      "clGetKernelWorkGroupInfo failed to return the maximum workgroup size for the kernel");
  ARM_COMPUTE_UNUSED(err);

  return result;
}

cl::NDRange CLKernelLibraryEx::default_ndrange() const
{
  //    GPUTarget   _target = get_target_from_device(_device);
  cl::Device device = cl::Device::getDefault();
  GPUTarget _target = get_target_from_device(device);
  cl::NDRange default_range;

  switch (_target)
  {
    case GPUTarget::MIDGARD:
    case GPUTarget::T600:
    case GPUTarget::T700:
    case GPUTarget::T800:
      default_range = cl::NDRange(128u, 1);
      break;
    default:
      default_range = cl::NullRange;
  }

  return default_range;
}

std::string CLKernelLibraryEx::get_device_version() { return _device.getInfo<CL_DEVICE_VERSION>(); }
