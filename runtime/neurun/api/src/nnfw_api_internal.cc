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
#include "exec/Execution.h"
#include "circle_loader.h"
#include "tflite_loader.h"
#include "json/json.h"
#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>
#include <util/ConfigSource.h>

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

static neurun::ir::Layout convertLayout(NNFW_LAYOUT layout)
{
  if (layout == NNFW_LAYOUT_CHANNELS_LAST)
  {
    return neurun::ir::Layout::NHWC;
  }
  else if (layout == NNFW_LAYOUT_CHANNELS_FIRST)
  {
    return neurun::ir::Layout::NCHW;
  }
  return neurun::ir::Layout::UNKNOWN;
}

nnfw_session::nnfw_session()
    : _graph{nullptr}, _execution{nullptr},
      _kernel_registry{std::make_shared<neurun::frontend::custom::KernelRegistry>()},
      _source{std::make_unique<neurun::util::GeneralConfigSource>()}
{
  // DO NOTHING
}

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
      _graph = neurun::tflite_loader::loadModel(model_file_path.c_str());
    }
    else if (model_type == "circle")
    {
      _graph = neurun::circle_loader::loadModel(model_file_path.c_str());
    }
    else
    {
      std::cerr << "Unsupported model type in MANIFEST" << std::endl;
      return NNFW_STATUS_ERROR;
    }
    _graph->bindKernelBuilder(_kernel_registry->getBuilder());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::prepare()
{
  if (!_graph || _graph->isBuildingPhase())
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

  _source->set("DELETE_CACHED_DATA", "1");

  try
  {
    // config_source setting
    using neurun::util::config_source;
    config_source(std::move(_source));

    auto compiler = std::make_unique<neurun::compiler::Compiler>(_graph);
    compiler->compile();
    std::shared_ptr<neurun::exec::IExecutor> executor;
    compiler->release(executor);
    _execution = std::make_shared<neurun::exec::Execution>(executor);
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
    _execution->setInput(neurun::ir::IOIndex(index), buffer, length);
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
    _execution->setOutput(neurun::ir::IOIndex(index), buffer, length);
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
    *number = _graph->getInputs().size();
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
    *number = _graph->getOutputs().size();
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
    _execution->setInputLayout(neurun::ir::IOIndex(index), convertLayout(layout));
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
    _execution->setOutputLayout(neurun::ir::IOIndex(index), convertLayout(layout));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output_layout : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

static NNFW_TYPE datatype_to_nnfw_dtype(neurun::ir::DataType dt)
{
  using neurun::ir::DataType;
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

NNFW_STATUS nnfw_session::apply_tensorinfo(uint32_t /*index*/, nnfw_tensorinfo /*ti*/)
{
  std::cerr << "Error: NYI" << std::endl;
  return NNFW_STATUS_ERROR;
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
    if (index >= _graph->getInputs().size())
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto opidx = _graph->getInputs().at(index);
    auto shape = _graph->operands().at(opidx).shape();
    ti->rank = shape.rank();
    for (int j = 0; j < ti->rank; ++j)
    {
      ti->dims[j] = shape.dim(j);
    }
    ti->dtype = datatype_to_nnfw_dtype(_graph->operands().at(opidx).typeInfo().type());
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
    if (index >= _graph->getOutputs().size())
    {
      std::cerr << "Error during nnfw_session::output_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto opidx = _graph->getOutputs().at(index);
    auto shape = _graph->operands().at(opidx).shape();
    ti->rank = shape.rank();
    for (int j = 0; j < ti->rank; ++j)
    {
      ti->dims[j] = shape.dim(j);
    }
    ti->dtype = datatype_to_nnfw_dtype(_graph->operands().at(opidx).typeInfo().type());
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
#define MAP_MACRO(CircleName, NeurunName) {#CircleName, "OP_BACKEND_" #NeurunName},

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
