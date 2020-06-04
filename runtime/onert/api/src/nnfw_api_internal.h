/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __API_NNFW_API_INTERNAL_H__
#define __API_NNFW_API_INTERNAL_H__

#include "nnfw.h"
#include "nnfw_dev.h"

#include <util/GeneralConfigSource.h>

#include <string>
#include <memory>

namespace onert
{
namespace frontend
{
namespace custom
{
class KernelRegistry;
}
} // namespace frontend
namespace exec
{
class Execution;
} // namespace exec
namespace ir
{
class Graph;
class Subgraphs;
} // namespace ir
namespace compiler
{
class Compiler;
} // namespace compiler
} // namespace onert

struct nnfw_session
{
private:
  enum class State
  {
    INITIALIZED,  //< Session is initialized and nothing has done to it
    MODEL_LOADED, //< Model is loaded
    PREPARED,     //< Prepared(compiled) for execution
  };

public:
  nnfw_session();
  ~nnfw_session();

  NNFW_STATUS load_model_from_file(const char *package_file_path);
  NNFW_STATUS prepare();
  NNFW_STATUS run();

  NNFW_STATUS set_input(uint32_t index, NNFW_TYPE type, const void *buffer, size_t length);
  NNFW_STATUS set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);

  NNFW_STATUS input_size(uint32_t *number);
  NNFW_STATUS output_size(uint32_t *number);

  NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout);

  NNFW_STATUS apply_tensorinfo(uint32_t index, nnfw_tensorinfo ti); // Will be deprecated
  NNFW_STATUS set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti);

  NNFW_STATUS input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS output_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);

  NNFW_STATUS register_custom_operation(const std::string &id, nnfw_custom_eval eval_func);

  NNFW_STATUS set_available_backends(const char *backends);
  NNFW_STATUS set_op_backend(const char *op, const char *backend);

  NNFW_STATUS set_config(const char *key, const char *value);
  NNFW_STATUS get_config(const char *key, char *value, size_t value_size);

private:
  onert::ir::Graph *primary_subgraph();
  bool isStateInitialized();
  bool isStateModelLoaded();
  bool isStatePrepared();

private:
  State _state{State::INITIALIZED};
  std::shared_ptr<onert::ir::Subgraphs> _subgraphs;
  std::unique_ptr<onert::compiler::Compiler> _compiler;
  std::shared_ptr<onert::exec::Execution> _execution;
  std::shared_ptr<onert::frontend::custom::KernelRegistry> _kernel_registry;

protected:
  std::unique_ptr<onert::util::GeneralConfigSource> _source;
};

#endif // __API_NNFW_API_INTERNAL_H__
