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
#include "exec/Execution.h"
#include "circle_loader.h"
#include "tflite_loader.h"
#include "json/json.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <util/ConfigSource.h>
#include <misc/string_helpers.h>

/*
 * API does not accept string argument longer than max length below
 */
#define MAX_BACKEND_NAME_LENGTH 32
#define MAX_OP_NAME_LENGTH 64

// Is null-terminating in length ?
static bool null_terminating(const char *str, uint32_t length)
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

static onert::ir::Layout convertLayout(NNFW_LAYOUT layout)
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

nnfw_session::nnfw_session()
    : _subgraphs{nullptr}, _execution{nullptr},
      _kernel_registry{std::make_shared<onert::frontend::custom::KernelRegistry>()},
      _source{std::make_unique<onert::util::GeneralConfigSource>()}
{
  // DO NOTHING
}

nnfw_session::~nnfw_session() = default;

NNFW_STATUS nnfw_session::load_model_from_file(const char *package_dir)
{
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
    Json::Value models = root["models"];
    Json::Value model_types = root["model-types"];

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

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::prepare()
{
  if (!_subgraphs || !primary_subgraph() || primary_subgraph()->isBuildingPhase())
  {
    std::cerr << "Error during model prepare : "
              << "prepare should be run after load_model" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  // NOTE. If users want to run prepare() more than one time, this could be removed.
  if (!_source || _execution)
  {
    std::cerr << "Error during model prepare : "
              << "prepare should be run once" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  // TODO : add additional setting routine(executor type, backend)
  // Note that we assume acl_cl backend

  set_config(onert::util::config::DELETE_CACHED_DATA, "1");

  try
  {
    // config_source setting
    using onert::util::config_source;
    config_source(std::move(_source));

    _subgraphs.reset();
    _compiler->compile();
    std::shared_ptr<onert::exec::ExecutorMap> executors;
    _compiler->release(executors);
    _execution = std::make_shared<onert::exec::Execution>(executors);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model prepare : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::run()
{
  if (!_execution)
  {
    std::cerr << "Error during nnfw_session::run : "
              << "run should be run after prepare" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _execution->execute();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::run : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_input(uint32_t index, NNFW_TYPE /*type*/, const void *buffer,
                                    size_t length)
{
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
  try
  {
    if (number == nullptr)
    {
      std::cerr << "Error during nnfw_session::input_size, number is null pointer." << std::endl;
      return NNFW_STATUS_ERROR;
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
  try
  {
    if (number == nullptr)
    {
      std::cerr << "Error during nnfw_session::output_size, number is null pointer." << std::endl;
      return NNFW_STATUS_ERROR;
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
    case DataType::QUANT8_ASYMM:
      return NNFW_TYPE_TENSOR_QUANT8_ASYMM;
    case DataType::BOOL8:
      return NNFW_TYPE_TENSOR_BOOL;
    case DataType::UINT8:
      return NNFW_TYPE_TENSOR_UINT8;
    case DataType::UINT32:
    case DataType::QUANT8_SYMM:
    default:
      std::cerr << "Error: Model has type that runtime API does not support." << std::endl;
      exit(-1);
  }
}

NNFW_STATUS nnfw_session::apply_tensorinfo(uint32_t index, nnfw_tensorinfo ti)
{
  // sanity check
  {
    if (!_subgraphs || !primary_subgraph() || primary_subgraph()->isBuildingPhase())
    {
      std::cerr << "Error during apply_tensorinfo : "
                << "prepare should be run after load_model" << std::endl;
      return NNFW_STATUS_ERROR;
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

  // when called before nnfw_session::prepare()
  if (!_execution)
  {
    // In this case, if we apply input shape in primary_subgraph, it will propagate after
    // compilation and excution
    auto ind = primary_subgraph()->getInputs().at(index);
    auto &input = primary_subgraph()->operands().at(ind);

    onert::ir::Shape new_shape(ti.rank);
    for (int32_t i = 0; i < ti.rank; i++)
      new_shape.dim(i) = ti.dims[i];

    // overwrite input shape with the shape from ti
    input.info().shape(new_shape);
  }
  else // when called after nnfw_session::prepare() but before excute()
  {
    onert::ir::Shape new_shape(ti.rank);
    for (int32_t i = 0; i < ti.rank; i++)
      new_shape.dim(i) = ti.dims[i];

    _execution->changeInputShape(onert::ir::IOIndex(index), new_shape);
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  try
  {
    if (ti == nullptr)
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, tensorinfo is null pointer."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    if (index >= primary_subgraph()->getInputs().size())
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto opidx = primary_subgraph()->getInputs().at(index);
    auto shape = primary_subgraph()->operands().at(opidx).shape();
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
  try
  {
    if (ti == nullptr)
    {
      std::cerr << "Error during nnfw_session::output_tensorinfo, tensorinfo is null pointer."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    if (index >= primary_subgraph()->getOutputs().size())
    {
      std::cerr << "Error during nnfw_session::output_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto opidx = primary_subgraph()->getOutputs().at(index);
    auto shape = primary_subgraph()->operands().at(opidx).shape();
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
#define MAP_MACRO(CircleName, OneRTName) {#CircleName, "OP_BACKEND_" #OneRTName},

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
  try
  {
    if (!backends || null_terminating(backends, MAX_BACKEND_NAME_LENGTH) == false)
    {
      return NNFW_STATUS_ERROR;
    }

    _source->set("BACKENDS", backends);
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
  try
  {
    if (!op || !null_terminating(op, MAX_OP_NAME_LENGTH) || !backend ||
        !null_terminating(backend, MAX_BACKEND_NAME_LENGTH))
    {
      return NNFW_STATUS_ERROR;
    }

    auto key = get_op_backend_string(op);

    if (key.empty())
    {
      return NNFW_STATUS_ERROR;
    }

    _source->set(key, backend);
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
  // The session must be in the state after model load
  if (!_compiler)
    return NNFW_STATUS_ERROR;

  auto &options = _compiler->options();

  using namespace onert::util;

  if (key == config::TRACE_FILEPATH)
  {
    options.trace_filepath = value;
  }
  else if (key == config::GRAPH_DOT_DUMP)
  {
    options.graph_dump_level = toInt(value);
  }
  else if (key == config::OP_SEQ_MAX_NODE)
  {
    options.op_seq_max_node = toInt(value);
  }
  else if (key == config::EXECUTOR)
  {
    options.executor = value;
  }
  else if (key == config::OP_BACKEND_ALLOPS)
  {
    options.manual_scheduler_options.backend_for_all = value;
  }
  else if (key == config::USE_SCHEDULER)
  {
    options.he_scheduler = toBool(value);
  }
  else if (key == config::PROFILING_MODE)
  {
    options.he_profiling_mode = toBool(value);
  }
  else if (key == config::DELETE_CACHED_DATA)
  {
    options.delete_cached_data = toBool(value);
  }
  else if (key == config::DISABLE_COMPILE)
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
  // The session must be in the state after model load
  if (!_compiler)
    return NNFW_STATUS_ERROR;

  auto &options = _compiler->options();

  auto check_boundary = [](size_t dest_size, std::string &src) {
    if (dest_size < src.length() + 1 /* for '\0' */)
    {
      std::cerr << "buffer is small to copy config value." << std::endl;
      return false;
    }
    return true;
  };

  if (key == onert::util::config::BACKENDS)
  {
    if (options.backend_list.size() == 0)
      return NNFW_STATUS_NO_ERROR; // no setting backend is not an error of get_config_str()

    auto str = nnfw::misc::join(options.backend_list.begin(), options.backend_list.end(), ";");

    if (!check_boundary(value_size, str))
      return NNFW_STATUS_ERROR;

    strncpy(value, str.c_str(), value_size);
  }
  else if (key == onert::util::config::EXECUTOR)
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
