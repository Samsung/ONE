/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_ODC_CODEGEN_LOADER_H__
#define __ONERT_ODC_CODEGEN_LOADER_H__

#include "odc/ICodegen.h"

#include <functional>
#include <memory>

namespace onert
{
namespace odc
{

/**
 * @brief Class to manage loading and unloading of dynamic library containing
 *        implementation of ICodegen interface.
 */
class CodegenLoader
{
public:
  /**
   * @brief Typedef for function pointer to destroy loaded library handle
   */
  using dlhandle_destroy_t = std::function<void(void *)>;
  /**
   * @brief Typedef for function pointer to create instance of ICodegen
   */
  using factory_t = ICodegen *(*)();
  /**
   * @brief Typedef for function pointer to destroy instance of ICodegen
   */
  using codegen_destory_t = void (*)(ICodegen *);

  /**
   * @brief Get singleton instance of CodegenLoader
   * @return Reference to singleton instance of CodegenLoader
   */
  static CodegenLoader &instance();

private:
  // cannot create instance of CodegenLoader outside of this class
  CodegenLoader() = default;
  CodegenLoader(CodegenLoader const &) = delete;
  CodegenLoader &operator=(CodegenLoader const &) = delete;
  ~CodegenLoader() = default;

public:
  /**
   * @brief   Load dynamic library containing implementation of ICodegen
   * @param[in] target Target backend name
   *                   This target string will be used to find a backend library.
   *                   The name of target backend library should follow the following rules:
   *                     'lib' + {backend extension} + '-gen' + {lib extension}
   *                   And the target string should be a name except 'lib' and {lib extension}.
   *                   For example, if the backend extension is 'aaa', the backend library name
   *                   should be 'libaaa-gen.so', and the target string should be 'aaa-gen'.
   */
  void loadLibrary(const char *target);
  /**
   * @brief  Unload dynamic library containing implementation of ICodegen
   */
  void unloadLibrary();
  /**
   * @brief   Get instance of ICodegen created through factory method
   * @return  Pointer to instance of ICodegen
   */
  ICodegen *get() const { return _codegen.get(); }

private:
  // Note: Keep handle to avoid svace warning of "handle lost without dlclose()"
  std::unique_ptr<void, dlhandle_destroy_t> _dlhandle;
  std::unique_ptr<ICodegen, codegen_destory_t> _codegen{nullptr, nullptr};
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_CODEGEN_LOADER_H__
