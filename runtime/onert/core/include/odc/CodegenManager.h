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

#ifndef __ONERT_ODC_CODEGEN_MANAGER_H__
#define __ONERT_ODC_CODEGEN_MANAGER_H__

#include <string>

namespace onert
{
namespace odc
{

enum class CodegenPreference
{
  CODEGEN_PREF_DEFAULT,
  // TODO Support Traffic and Cycle codegen preference
  CODEGEN_PREF_PERFORMANCE_FIRST,
  CODEGEN_PREF_MEMORY_FIRST,
  CODEGEN_PREF_COMPILE_TIME_FIRST,
};

class CodegenManager
{
public:
  // Non-copyable
  CodegenManager(const std::string &model_path) : _model_path(model_path) {}
  CodegenManager(CodegenManager const &) = delete;
  CodegenManager &operator=(CodegenManager const &) = delete;

public:
  /**
   * @brief Set model path to export compiled model
   *
   * @param model_path  Model path to export compiled model
   */
  void exportModelPath(const std::string &model_path) { _export_model_path = model_path; }

  /**
   * @brief Get model path to export compiled model
   *
   * @return Model path to export compiled model
   */
  const std::string &exportModelPath() const { return _export_model_path; }

  /**
   * @brief Execute code generator
   *
   * @param target  Target backend name
   *                This target string will be used to find a backend library.
   *                The name of target backend library should follow the following rules:
   *                  'lib' + {backend extension} + '-gen' + {lib extension}
   *                And the target string should be a name except 'lib' and {lib extension}.
   *                For example, if the backend extension is 'aaa', the backend library name
   *                should be 'libaaa-gen.so', and the target string should be 'aaa-gen'.
   * @param pref    @c CodegenPreference Codegen preference
   */
  bool codegen(const char *target, CodegenPreference pref);

private:
  std::string _model_path = "";
  std::string _export_model_path = "";
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_CODEGEN_MANAGER_H__
