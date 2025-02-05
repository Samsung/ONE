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

/**
 * @file      CLKernelLibraryEx.h
 * @ingroup   COM_AI_RUNTIME
 * @brief     This file is a cloned version of CLKernelLibrary.h in ACL. This file defines
 *            an interface for CLKernelLibrary.cpp which adds more OpenCL kernels on top of ACL.
 */

#ifndef __ARM_COMPUTE_CLKERNELLIBRARY_EX_H__
#define __ARM_COMPUTE_CLKERNELLIBRARY_EX_H__

#include "arm_compute/core/CL/OpenCL.h"

#include <map>
#include <set>
#include <string>
#include <utility>

namespace arm_compute
{

/**
 * @brief Class to build OpenCL kernels added from nnfw
 * */
class CLKernelLibraryEx
{
  using StringSet = std::set<std::string>;

private:
  /**
   * @brief Construct a new CLKernelLibraryEx object
   */
  CLKernelLibraryEx();

public:
  /**
   * @brief Prevent instances of this class from being copied.
   */
  CLKernelLibraryEx(const CLKernelLibraryEx &) = delete;

  /**
   * @brief Prevent instances of this class from being copied.
   */
  const CLKernelLibraryEx &operator=(const CLKernelLibraryEx &) = delete;

  /**
   * @brief Get the KernelLibrary singleton.
   * @return The KernelLibrary instance
   */
  static CLKernelLibraryEx &get();

  /**
   * @brief Initialise the kernel library.
   * @param[in] kernel_path Path of the directory from which kernel sources are loaded.
   * @param[in] context     CL context used to create programs.
   * @param[in] device      CL device for which the programs are created.
   * @return N/A
   */
  void init(std::string kernel_path, cl::Context context, cl::Device device)
  {
    _kernel_path = std::move(kernel_path);
    _context = std::move(context);
    _device = std::move(device);
  }

  /**
   * @brief Set the path that the kernels reside in.
   * @param[in] kernel_path Path of the directory from which kernel sources are loaded.
   * @return N/A
   */
  void set_kernel_path(const std::string &kernel_path) { _kernel_path = kernel_path; };

  /**
   * @brief Get the path that the kernels reside in.
   * @return the path of kernel files
   */
  std::string get_kernel_path() { return _kernel_path; };

  /**
   * @brief Get the source of the selected program.
   * @param[in] program_name Program name.
   * @return Source of the selected program.
   */
  std::string get_program_source(const std::string &program_name);

  /**
   * @brief Set the CL context used to create programs.
   * @note Setting the context also resets the device to the
   *       first one available in the new context.
   * @param[in] context A CL context.
   * @return N/A
   */
  void set_context(cl::Context context)
  {
    _context = std::move(context);
    if (_context.get() == nullptr)
    {
      _device = cl::Device();
    }
    else
    {
      const auto cl_devices = _context.getInfo<CL_CONTEXT_DEVICES>();

      if (cl_devices.empty())
      {
        _device = cl::Device();
      }
      else
      {
        _device = cl_devices[0];
      }
    }
  }

  /**
   * @brief Return associated CL context.
   * @return A CL context.
   */
  cl::Context &context() { return _context; }

  /**
   * @brief Set the CL device for which the programs are created.
   * @param[in] device A CL device.
   * @return N/A
   */
  void set_device(cl::Device device) { _device = std::move(device); }

  /**
   * @brief Gets the CL device for which the programs are created.
   * @return A CL device.
   */
  cl::Device &get_device() { return _device; }

  /**
   * @brief Return the device version
   * @return The content of CL_DEVICE_VERSION
   */
  std::string get_device_version();

  /**
   * @brief Create a kernel from the kernel library.
   * @param[in] kernel_name       Kernel name.
   * @param[in] build_options_set Kernel build options as a set.
   * @return The created kernel.
   */
  Kernel create_kernel(const std::string &kernel_name,
                       const StringSet &build_options_set = {}) const;

  /**
   * @brief Find the maximum number of local work items in a workgroup can be supported for the
   * kernel.
   * @param[in] kernel       kernel object
   */

  size_t max_local_workgroup_size(const cl::Kernel &kernel) const;
  /**
   * @brief Return the default NDRange for the device.
   * @return default NDRangeof the device
   */
  cl::NDRange default_ndrange() const;

  /**
   * @brief Clear the library's cache of binary programs
   * @return N/A
   */
  void clear_programs_cache()
  {
    _programs_map.clear();
    _built_programs_map.clear();
  }

  /**
   * @brief Access the cache of built OpenCL programs
   * @return program map data structure of which key is name of kernel and value is
   *         kerel source name. (*.cl)
   */
  const std::map<std::string, cl::Program> &get_built_programs() const
  {
    return _built_programs_map;
  }

  /**
   * @brief Add a new built program to the cache
   * @param[in] built_program_name Name of the program
   * @param[in] program            Built program to add to the cache
   * @return N/A
   */
  void add_built_program(const std::string &built_program_name, cl::Program program);

  /**
   * @brief Returns true if FP16 is supported by the CL device
   * @return true if the CL device supports FP16
   */
  bool fp16_supported() const;

  /**
   * @brief Returns true if int64_base_atomics extension is supported by the CL device
   * @return true if the CL device supports int64_base_atomics extension
   */
  bool int64_base_atomics_supported() const;

private:
  /**
   * @brief Load program and its dependencies.
   * @param[in] program_name Name of the program to load.
   */
  const Program &load_program(const std::string &program_name) const;
  /**
   * @brief Concatenates contents of a set into a single string.
   * @param[in] s Input set to concatenate.
   * @return Concatenated string.
   */
  std::string stringify_set(const StringSet &s) const;

  cl::Context _context;     /**< Underlying CL context. */
  cl::Device _device;       /**< Underlying CL device. */
  std::string _kernel_path; /**< Path to the kernels folder. */
  mutable std::map<std::string, const Program>
    _programs_map; /**< Map with all already loaded program data. */
  mutable std::map<std::string, cl::Program>
    _built_programs_map; /**< Map with all already built program data. */
  static const std::map<std::string, std::string>
    _kernel_program_map; /**< Map that associates kernel names with programs. */
  static const std::map<std::string, std::string>
    _program_source_map; /**< Contains sources for all programs.
                           Used for compile-time kernel inclusion. >*/
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLKERNELLIBRARY_EX_H__ */
