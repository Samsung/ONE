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

#include "nnfw_api_internal.h"
#include "CustomKernelRegistry.h"
#include "compiler/Compiler.h"
#include "util/ConfigSource.h"
#include "util/Exceptions.h"
#include "exec/Execution.h"
#include "circle_loader.h"
#include "tflite_loader.h"
#include "json/json.h"
#include "ir/OpCode.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <misc/string_helpers.h>

/*
 * API does not accept string argument longer than max length below
 */
#define MAX_BACKEND_NAME_LENGTH 32
#define MAX_OP_NAME_LENGTH 64
#define MAX_PATH_LENGTH 1024
#define MAX_TENSOR_NAME_LENGTH 64

namespace
{

// Is null-terminating in length ?
bool null_terminating(const char *str, uint32_t length)
{
  for (uint32_t i = 0; i < length; i++)
  {
    if (*(str + i) == '\0')
    {
      return true;
    }
  }
  return false;
}

onert::ir::Layout convertLayout(NNFW_LAYOUT layout)
{
  if (layout == NNFW_LAYOUT_CHANNELS_LAST)
  {
    return onert::ir::Layout::NHWC;
  }
  else if (layout == NNFW_LAYOUT_CHANNELS_FIRST)
  {
    return onert::ir::Layout::NCHW;
  }
  return onert::ir::Layout::UNKNOWN;
}

NNFW_STATUS getTensorIndexImpl(const onert::ir::Graph &graph, const char *tensorname,
                               uint32_t *index, bool is_input)
{
  if (!tensorname || !index)
    return NNFW_STATUS_UNEXPECTED_NULL;

  if (!null_terminating(tensorname, MAX_TENSOR_NAME_LENGTH))
  {
    std::cerr << "nnpackage path is too long" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  auto ind_found = is_input ? graph.getInputIndex(tensorname) : graph.getOutputIndex(tensorname);

  if (ind_found.undefined())
  {
    // Not found
    return NNFW_STATUS_ERROR;
  }
  else
  {
    *index = ind_found.value();
    return NNFW_STATUS_NO_ERROR;
  }
}

} // namespace

nnfw_session::nnfw_session()
    : _subgraphs{nullptr}, _execution{nullptr},
      _kernel_registry{std::make_shared<onert::frontend::custom::KernelRegistry>()}
{
  // DO NOTHING
}

nnfw_session::~nnfw_session() = default;

NNFW_STATUS nnfw_session::load_circle_from_buffer(uint8_t *buffer, size_t size)
{
  if (!isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (!buffer)
    return NNFW_STATUS_UNEXPECTED_NULL;

  if (size == 0)
    return NNFW_STATUS_ERROR;

  try
  {
    _subgraphs = onert::circle_loader::loadModel(buffer, size);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _compiler = std::make_unique<onert::compiler::Compiler>(_subgraphs);

  _state = State::MODEL_LOADED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::load_model_from_file(const char *package_dir)
{
  if (!isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (!package_dir)
  {
    std::cerr << "package_dir is null." << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!null_terminating(package_dir, MAX_PATH_LENGTH))
  {
    std::cerr << "nnpackage path is too long" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  // TODO : add support for zipped package file load
  DIR *dir;
  if (!(dir = opendir(package_dir)))
  {
    std::cerr << "invalid nnpackge directory: " << package_dir << std::endl;
    return NNFW_STATUS_ERROR;
  }
  closedir(dir);

  try
  {
    std::string manifest_file_name(package_dir);
    manifest_file_name += "/metadata/MANIFEST";
    std::ifstream mfs(manifest_file_name);

    // extract the filename of the first(index 0) model
    // e.g. In MANIFEST file, { "models" : [ "firstmodel.tflite", "2nd.tflite" ] }
    Json::Value root;
    mfs >> root;
    const Json::Value &models = root["models"];
    const Json::Value &model_types = root["model-types"];

    auto model_file_path = package_dir + std::string("/") + models[0].asString(); // first model
    auto model_type = model_types[0].asString(); // first model's type
    if (model_type == "tflite")
    {
      _subgraphs = onert::tflite_loader::loadModel(model_file_path.c_str());
    }
    else if (model_type == "circle")
    {
      _subgraphs = onert::circle_loader::loadModel(model_file_path.c_str());
    }
    else
    {
      std::cerr << "Unsupported model type in MANIFEST" << std::endl;
      return NNFW_STATUS_ERROR;
    }
    _subgraphs->primary()->bindKernelBuilder(_kernel_registry->getBuilder());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _compiler = std::make_unique<onert::compiler::Compiler>(_subgraphs);

  _state = State::MODEL_LOADED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::prepare()
{
  // NOTE. If users want to run prepare() more than one time, this could be removed.
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during model prepare : ";
    if (isStateInitialized())
    {
      std::cerr << "prepare should be run once";
    }
    else
    {
      std::cerr << "invalid state";
    }
    std::cerr << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (!_subgraphs || !primary_subgraph() || primary_subgraph()->isBuildingPhase())
  {
    std::cerr << "Error during model prepare : "
              << "prepare should be run after load_model" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _subgraphs.reset();
    std::shared_ptr<onert::exec::ExecutorMap> executors = _compiler->compile();
    _execution = std::make_shared<onert::exec::Execution>(executors);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model prepare : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::PREPARED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::run()
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run : "
              << "run should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    _execution->execute();
  }
  catch (const onert::InsufficientBufferSizeException &e)
  {
    // Currently insufficient buffer always means output buffer.
    std::cerr << "Error during nnfw_session::run : " << e.what() << std::endl;
    return NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::run : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::FINISHED_RUN;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::run_async()
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run_async : "
              << "run_async should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  _execution->startExecute();

  _state = State::RUNNING;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::await()
{
  if (!isStateRunning())
  {
    std::cerr << "Error during nnfw_session::run_await : "
              << "run_await should be run after run_async" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _execution->waitFinish();

  _state = State::FINISHED_RUN;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_input(uint32_t index, NNFW_TYPE /*type*/, const void *buffer,
                                    size_t length)
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_input : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (!buffer && length != 0)
  {
    std::cerr
        << "Error during nnfw_session::set_input : given buffer is NULL but the length is not 0"
        << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _execution->setInput(onert::ir::IOIndex(index), buffer, length);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_input : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_output(uint32_t index, NNFW_TYPE /*type*/, void *buffer,
                                     size_t length)
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_output : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (!buffer && length != 0)
  {
    std::cerr
        << "Error during nnfw_session::set_output : given buffer is NULL but the length is not 0"
        << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _execution->setOutput(onert::ir::IOIndex(index), buffer, length);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::input_size(uint32_t *number)
{
  if (isStateInitialized()) // Model is not loaded
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (number == nullptr)
    {
      std::cerr << "Error during nnfw_session::input_size, number is null pointer." << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }
    *number = primary_subgraph()->getInputs().size();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::input_size : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::output_size(uint32_t *number)
{
  if (isStateInitialized()) // Model is not loaded
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (number == nullptr)
    {
      std::cerr << "Error during nnfw_session::output_size, number is null pointer." << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }
    *number = primary_subgraph()->getOutputs().size();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::output_size" << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_input_layout(uint32_t index, NNFW_LAYOUT layout)
{
  try
  {
    if (layout != NNFW_LAYOUT_NONE && layout != NNFW_LAYOUT_CHANNELS_FIRST &&
        layout != NNFW_LAYOUT_CHANNELS_LAST)
    {
      std::cerr << "Error during nnfw_session::set_input_layout, not supported layout" << std::endl;
      return NNFW_STATUS_ERROR;
    }
    _execution->setInputLayout(onert::ir::IOIndex(index), convertLayout(layout));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_input_layout : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_output_layout(uint32_t index, NNFW_LAYOUT layout)
{
  try
  {
    if (layout != NNFW_LAYOUT_NONE && layout != NNFW_LAYOUT_CHANNELS_FIRST &&
        layout != NNFW_LAYOUT_CHANNELS_LAST)
    {
      std::cerr << "Error during nnfw_session::set_output_layout, not supported layout"
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    _execution->setOutputLayout(onert::ir::IOIndex(index), convertLayout(layout));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output_layout : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

static NNFW_TYPE datatype_to_nnfw_dtype(onert::ir::DataType dt)
{
  using onert::ir::DataType;
  switch (dt)
  {
    case DataType::FLOAT32:
      return NNFW_TYPE_TENSOR_FLOAT32;
    case DataType::INT32:
      return NNFW_TYPE_TENSOR_INT32;
    case DataType::QUANT_UINT8_ASYMM:
      return NNFW_TYPE_TENSOR_QUANT8_ASYMM;
    case DataType::BOOL8:
      return NNFW_TYPE_TENSOR_BOOL;
    case DataType::UINT8:
      return NNFW_TYPE_TENSOR_UINT8;
    case DataType::INT64:
      return NNFW_TYPE_TENSOR_INT64;
    case DataType::QUANT_INT8_ASYMM:
      return NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED;
    case DataType::UINT32:
    case DataType::QUANT_INT8_SYMM:
    default:
      throw std::runtime_error("Error: Model has type that runtime API does not support.");
  }
}

NNFW_STATUS nnfw_session::apply_tensorinfo(uint32_t index, nnfw_tensorinfo ti)
{
  // sanity check
  {
    if (isStateInitialized())
    {
      std::cerr << "Error during set_input_tensorinfo : should be run after load_model"
                << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    if (ti.rank <= 0 || ti.rank > NNFW_MAX_RANK)
    {
      std::cerr << "unsupported rank: " << ti.rank << std::endl;
      return NNFW_STATUS_ERROR;
    }

    for (int32_t i = 0; i < ti.rank; ++i)
    {
      if (ti.dims[i] <= 0)
      {
        std::cerr << "dim must be positive integer but was " << ti.dims[i] << std::endl;
        return NNFW_STATUS_ERROR;
      }
    }
  }

  onert::ir::Shape new_shape(ti.rank);
  for (int32_t i = 0; i < ti.rank; i++)
    new_shape.dim(i) = ti.dims[i];

  if (!isStatePreparedOrFinishedRun())
  {
    // In this case, if we apply input shape in primary_subgraph, it will propagate after
    // compilation and excution
    auto ind = primary_subgraph()->getInputs().at(index);
    auto &input = primary_subgraph()->operands().at(ind);

    // overwrite input shape with the shape from ti
    input.info().shape(new_shape);
  }
  else // when called after nnfw_session::prepare()
  {
    _execution->changeInputShape(onert::ir::IOIndex(index), new_shape);
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti)
{
  nnfw_tensorinfo ti_copy = *ti;
  return apply_tensorinfo(index, ti_copy);
}

NNFW_STATUS nnfw_session::input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  if (isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (ti == nullptr)
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, tensorinfo is null pointer."
                << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }
    if (index >= primary_subgraph()->getInputs().size())
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto opidx = primary_subgraph()->getInputs().at(index);
    auto shape = primary_subgraph()->operands().at(opidx).shape();
    if (isStatePreparedOrFinishedRun())
      shape = _execution->getInputShape(onert::ir::IOIndex{index});
    ti->rank = shape.rank();
    for (int j = 0; j < ti->rank; ++j)
    {
      ti->dims[j] = shape.dim(j);
    }
    ti->dtype = datatype_to_nnfw_dtype(primary_subgraph()->operands().at(opidx).typeInfo().type());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::input_tensorinfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::output_tensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  if (isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (ti == nullptr)
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo, tensorinfo is null pointer."
              << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (index >= primary_subgraph()->getOutputs().size())
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo, index is out of range."
              << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    auto opidx = primary_subgraph()->getOutputs().at(index);
    auto shape = primary_subgraph()->operands().at(opidx).shape();
    // If it is called after `nnfw_run` then get the shape from Execution, not from the graph
    if (isStateFinishedRun())
      shape = _execution->getOutputShape(onert::ir::IOIndex{index});
    ti->rank = shape.rank();
    for (int j = 0; j < ti->rank; ++j)
    {
      ti->dims[j] = shape.dim(j);
    }
    ti->dtype = datatype_to_nnfw_dtype(primary_subgraph()->operands().at(opidx).typeInfo().type());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}
NNFW_STATUS nnfw_session::register_custom_operation(const std::string &id,
                                                    nnfw_custom_eval eval_func)
{
  _kernel_registry->registerKernel(id, eval_func);
  return NNFW_STATUS_NO_ERROR;
}

static std::string get_op_backend_string(std::string op)
{
#define MAP_MACRO(CircleName, OneRTName) {#CircleName, #OneRTName},

  static std::unordered_map<std::string, std::string> operation_map = {
#include "OpMap.lst"
  };

#undef MAP_MACRO

  auto n = operation_map.find(op);

  if (n == operation_map.end())
  {
    // this return value is handled by a caller to return error code
    return std::string("");
  }
  else
  {
    return n->second;
  }
}

NNFW_STATUS nnfw_session::set_available_backends(const char *backends)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (!backends)
      return NNFW_STATUS_UNEXPECTED_NULL;
    if (null_terminating(backends, MAX_BACKEND_NAME_LENGTH) == false)
      return NNFW_STATUS_ERROR;

    auto &options = _compiler->options();

    using namespace onert::util;

    options.backend_list = nnfw::misc::split(std::string{backends}, ';');
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_available_backends : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_op_backend(const char *op, const char *backend)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (!op || !backend)
      return NNFW_STATUS_UNEXPECTED_NULL;
    if (!null_terminating(op, MAX_OP_NAME_LENGTH) ||
        !null_terminating(backend, MAX_BACKEND_NAME_LENGTH))
      return NNFW_STATUS_ERROR;

    auto key = get_op_backend_string(op);

    if (key.empty())
    {
      return NNFW_STATUS_ERROR;
    }

    auto &opcode_to_backend = _compiler->options().manual_scheduler_options.opcode_to_backend;
    opcode_to_backend.emplace(onert::ir::toOpCode(key), backend);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_op_backend : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_config(const char *key, const char *value)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = _compiler->options();

  using namespace onert::util;

  const std::string skey = key;

  if (skey == config::TRACE_FILEPATH)
  {
    options.trace_filepath = value;
  }
  else if (skey == config::GRAPH_DOT_DUMP)
  {
    options.graph_dump_level = toInt(value);
  }
  else if (skey == config::OP_SEQ_MAX_NODE)
  {
    options.op_seq_max_node = toInt(value);
  }
  else if (skey == config::EXECUTOR)
  {
    options.executor = value;
  }
  else if (skey == config::OP_BACKEND_ALLOPS)
  {
    options.manual_scheduler_options.backend_for_all = value;
  }
  else if (skey == config::USE_SCHEDULER)
  {
    options.he_scheduler = toBool(value);
  }
  else if (skey == config::PROFILING_MODE)
  {
    options.he_profiling_mode = toBool(value);
  }
  else if (skey == config::DISABLE_COMPILE)
  {
    options.disable_compile = toBool(value);
  }
  else
  {
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

onert::ir::Graph *nnfw_session::primary_subgraph()
{
  if (_subgraphs)
  {
    assert(!_execution);
    return _subgraphs->primary().get();
  }
  else
  {
    assert(_execution);
    // TODO Remove const_cast
    // We assumed the graph will not change after compilation, but shape could change
    return const_cast<onert::ir::Graph *>(&_execution->primary_subgraph());
  }
}

NNFW_STATUS nnfw_session::get_config(const char *key, char *value, size_t value_size)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = _compiler->options();

  auto check_boundary = [](size_t dest_size, std::string &src) {
    if (dest_size < src.length() + 1 /* for '\0' */)
    {
      std::cerr << "buffer is small to copy config value." << std::endl;
      return false;
    }
    return true;
  };

  const std::string skey = key;

  if (skey == onert::util::config::BACKENDS)
  {
    if (options.backend_list.size() == 0)
      return NNFW_STATUS_NO_ERROR; // no setting backend is not an error of get_config_str()

    auto str = nnfw::misc::join(options.backend_list.begin(), options.backend_list.end(), ";");

    if (!check_boundary(value_size, str))
      return NNFW_STATUS_ERROR;

    strncpy(value, str.c_str(), value_size);
  }
  else if (skey == onert::util::config::EXECUTOR)
  {
    if (!check_boundary(value_size, options.executor))
      return NNFW_STATUS_ERROR;

    strncpy(value, options.executor.c_str(), options.executor.length());
  }
  else
  {
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

bool nnfw_session::isStateInitialized()
{
  if (_state == State::INITIALIZED)
  {
    assert(!_subgraphs);
    assert(!_compiler);
    assert(!_execution);
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStateModelLoaded()
{
  if (_state == State::MODEL_LOADED)
  {
    assert(_subgraphs);
    assert(_compiler);
    assert(!_execution);
    assert(!primary_subgraph()->isBuildingPhase());
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStatePrepared()
{
  if (_state == State::PREPARED)
  {
    assert(!_subgraphs);
    assert(_compiler);
    assert(_execution);
    assert(!primary_subgraph()->isBuildingPhase());
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStateRunning()
{
  if (_state == State::RUNNING)
  {
    assert(!_subgraphs);
    assert(_compiler);
    assert(_execution);
    assert(!primary_subgraph()->isBuildingPhase());
    return true;
  }
  return false;
}

bool nnfw_session::isStateFinishedRun()
{
  if (_state == State::FINISHED_RUN)
  {
    assert(!_subgraphs);
    assert(_compiler);
    assert(_execution);
    assert(!primary_subgraph()->isBuildingPhase());
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStatePreparedOrFinishedRun()
{
  return isStatePrepared() || isStateFinishedRun();
}

NNFW_STATUS nnfw_session::input_tensorindex(const char *tensorname, uint32_t *index)
{
  return getTensorIndexImpl(*primary_subgraph(), tensorname, index, true);
}

NNFW_STATUS nnfw_session::output_tensorindex(const char *tensorname, uint32_t *index)
{
  return getTensorIndexImpl(*primary_subgraph(), tensorname, index, false);
}
