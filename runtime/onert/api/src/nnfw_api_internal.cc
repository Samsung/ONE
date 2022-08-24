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
#include "util/logging.h"
#include "exec/Execution.h"
#include "circle_loader.h"
#include "tflite_loader.h"
#include "trix_loader.h"
#include "json/json.h"
#include "ir/NNPkg.h"
#include "ir/OpCode.h"
#include "util/TracingCtx.h"

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

std::string trim(const std::string &value)
{
  std::string whitespace = " \t";
  auto begin = value.find_first_not_of(whitespace);
  if (begin == std::string::npos)
    return ""; // no content

  auto end = value.find_last_not_of(whitespace);
  auto range = end - begin + 1;
  return value.substr(begin, range);
}

bool loadConfigure(const std::string cfgfile, onert::util::CfgKeyValues &keyValues)
{
  std::ifstream ifs(cfgfile);
  if (ifs.is_open())
  {
    std::string line;
    while (std::getline(ifs, line))
    {
      auto cmtpos = line.find('#');
      if (cmtpos != std::string::npos)
      {
        line = line.substr(0, cmtpos);
      }
      std::istringstream isline(line);
      std::string key;
      if (std::getline(isline, key, '='))
      {
        std::string value;
        if (std::getline(isline, value))
        {
          key = trim(key);
          keyValues[key] = trim(value);
        }
      }
    }
    ifs.close();
    return true;
  }
  return false;
}

NNFW_TYPE datatype_to_nnfw_dtype(onert::ir::DataType dt)
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
    case DataType::QUANT_INT16_SYMM:
      return NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED;
    case DataType::UINT32:
    case DataType::QUANT_INT8_SYMM:
    default:
      throw std::runtime_error("Error: Model has type that runtime API does not support.");
  }
}

void fillTensorInfo(nnfw_tensorinfo *ti, const onert::ir::Shape &shape,
                    const onert::ir::DataType &dtype)
{
  ti->rank = shape.rank();
  for (int j = 0; j < ti->rank; ++j)
  {
    ti->dims[j] = shape.dim(j);
  }
  ti->dtype = datatype_to_nnfw_dtype(dtype);
}

std::unique_ptr<onert::ir::Model> loadModel(const std::string filename,
                                            const std::string model_type)
{
  if (model_type == "tflite")
    return onert::tflite_loader::loadModel(filename.c_str());
  if (model_type == "circle")
    return onert::circle_loader::loadModel(filename.c_str());
  if (model_type == "tvn")
    return onert::trix_loader::loadModel(filename.c_str());

  std::cerr << "Unsupported model type" << std::endl;
  return std::unique_ptr<onert::ir::Model>(nullptr);
}

} // namespace

nnfw_session::nnfw_session()
  : _nnpkg{nullptr}, _coptions{}, _compiler_artifact{nullptr}, _execution{nullptr},
    _kernel_registry{nullptr}
{
  // DO NOTHING
}

NNFW_STATUS nnfw_session::create(nnfw_session **session)
{
  if (session == nullptr)
    return NNFW_STATUS_UNEXPECTED_NULL;

  // Create session
  *session = new (std::nothrow) nnfw_session();
  if (*session == nullptr)
  {
    std::cerr << "Error during session creation" << std::endl;
    return NNFW_STATUS_OUT_OF_MEMORY;
  }

  // Initialize fields
  try
  {
    (*session)->_kernel_registry = std::make_shared<onert::api::CustomKernelRegistry>();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during session initialization : " << e.what() << std::endl;
    delete *session;
    *session = nullptr;

    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
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
    auto model = onert::circle_loader::loadModel(buffer, size);
    _nnpkg = std::make_shared<onert::ir::NNPkg>(std::move(model));
    _coptions.push_back(onert::compiler::CompilerOptions::fromGlobalConfig());
    _state = State::MODEL_LOADED;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::load_model_from_modelfile(const char *model_file_path)
{
  if (!isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (!model_file_path)
  {
    std::cerr << "Model file path is null." << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  std::string filename{model_file_path};
  // TODO: Use std::filesystem::path when we can use c++17.
  auto dotidx = filename.find_last_of('.');
  if (dotidx == std::string::npos)
  {
    std::cerr << "Invalid model file path. Please use file with extension." << std::endl;
    return NNFW_STATUS_ERROR;
  }
  std::string model_type = filename.substr(dotidx + 1); // + 1 to exclude dot
  try
  {
    auto model = loadModel(filename, model_type);
    if (model == nullptr)
      return NNFW_STATUS_ERROR;
    _nnpkg = std::make_shared<onert::ir::NNPkg>(std::move(model));
    _coptions.push_back(onert::compiler::CompilerOptions::fromGlobalConfig());
    _state = State::MODEL_LOADED;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::load_model_from_nnpackage(const char *package_dir)
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
    std::string package_path(package_dir);
    std::string manifest_file_name = package_path + "/metadata/MANIFEST";
    std::ifstream mfs(manifest_file_name);

    _package_file_path = package_path;
    // extract the filename of the first(index 0) model
    // e.g. In MANIFEST file, { "models" : [ "firstmodel.tflite", "2nd.tflite" ] }
    Json::Value root;
    mfs >> root;
    const Json::Value &models = root["models"];
    const Json::Value &model_types = root["model-types"];
    const Json::Value &configs = root["configs"];

    if (!configs.empty() && !configs[0].empty())
    {
      auto filepath = package_path + std::string("/metadata/") + configs[0].asString();

      onert::util::CfgKeyValues keyValues;
      if (loadConfigure(filepath, keyValues))
      {
        onert::util::setConfigKeyValues(keyValues);
      }
    }
    _nnpkg = std::make_shared<onert::ir::NNPkg>();
    for (uint32_t i = 0; i < models.size(); ++i)
    {
      auto model_file_path = package_path + std::string("/") + models[i].asString();
      auto model_type = model_types[i].asString();
      auto model = loadModel(model_file_path, model_type);
      if (model == nullptr)
        return NNFW_STATUS_ERROR;
      model->primary_subgraph()->bindKernelBuilder(_kernel_registry->getBuilder());
      _nnpkg->push(onert::ir::ModelIndex{i}, std::move(model));
      _coptions.push_back(onert::compiler::CompilerOptions::fromGlobalConfig());
    }

    auto toIODesc = [](std::string str) {
      auto indices = nnfw::misc::split(str, ':');
      if (indices.size() != 3)
      {
        std::cerr << "IODesc should be 3-tuple." << std::endl;
        return onert::ir::IODesc{};
      }
      auto model_idx = static_cast<uint32_t>(std::stoi(indices.at(0)));
      auto subgraph_idx = static_cast<uint32_t>(std::stoi(indices.at(1)));
      auto operand_idx = static_cast<uint32_t>(std::stoi(indices.at(2)));
      return onert::ir::IODesc{model_idx, subgraph_idx, operand_idx};
    };
    // read pkg-inputs and pkg-outputs
    const Json::Value &pkg_inputs = root["pkg-inputs"];
    for (uint32_t i = 0; i < pkg_inputs.size(); ++i)
      _nnpkg->addInput(toIODesc(pkg_inputs[i].asString()));
    const Json::Value &pkg_outputs = root["pkg-outputs"];
    for (uint32_t i = 0; i < pkg_outputs.size(); ++i)
      _nnpkg->addOutput(toIODesc(pkg_outputs[i].asString()));
    // read model-connect
    const Json::Value &fromtos = root["model-connect"];
    for (uint32_t i = 0; i < fromtos.size(); ++i)
    {
      const Json::Value &tos = fromtos[i]["to"];
      for (uint32_t j = 0; j < tos.size(); ++j)
        _nnpkg->addEdge(toIODesc(fromtos[i]["from"].asString()), toIODesc(tos[j].asString()));
    }
    _state = State::MODEL_LOADED;
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

  try
  {
    // TODO: Compile all models in case of multiple models
    if (_nnpkg->model_count() > 1)
    {
      std::cerr << "Error during model prepare : multiple models are not supported yet."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }
    auto model = _nnpkg->primary_model();
    auto compiler = std::make_unique<onert::compiler::Compiler>(model, *_coptions[0]);
    _nnpkg.reset();
    _compiler_artifact = compiler->compile();
    _execution = std::make_unique<onert::exec::Execution>(_compiler_artifact->_executors);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model prepare : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::PREPARED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::prepare_pipeline(const char *map_file_path)
{
  // NOTE. If users want to run prepare_pipeline() more than one time, this could be removed.
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during model prepare pipeline : ";
    if (isStateInitialized())
    {
      std::cerr << "prepare_pipeline should be run once";
    }
    else
    {
      std::cerr << "invalid state";
    }
    std::cerr << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    auto model = _nnpkg->primary_model();
    auto compiler = std::make_unique<onert::compiler::Compiler>(model, *_coptions[0]);
    _nnpkg.reset();
    auto artifacts = compiler->compile(_package_file_path.c_str(), map_file_path);

    for (auto it = artifacts.begin(); it != artifacts.end(); ++it)
    {
      _executions.push_back(std::make_shared<onert::exec::Execution>(it->get()->_executors));
    }
    make_dependency();
    _threads.resize(_executions.size());
    for (uint32_t i = 0; i < _threads.size(); i++)
    {
      _threads[i] = std::thread(&onert::exec::Execution::runInference, _executions[i].get());
    }
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

  if (!_executions.empty())
  {
    std::cerr << "Error during nnfw_session::run : not supported for pipeline run" << std::endl;
    return NNFW_STATUS_ERROR;
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

  if (!_executions.empty())
  {
    std::cerr << "Error during nnfw_session::run_async : not supported for pipeline run"
              << std::endl;
    return NNFW_STATUS_ERROR;
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

  if (!_executions.empty())
  {
    std::cerr << "Error during nnfw_session::await : not supported for pipeline run" << std::endl;
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

  if (!_executions.empty())
  {
    std::cerr << "Error during nnfw_session::set_input : not supported for pipeline run"
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

  if (!_executions.empty())
  {
    std::cerr << "Error during nnfw_session::set_output : not supported for pipeline run"
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
    if (_execution)
    {
      _execution->setInputLayout(onert::ir::IOIndex(index), convertLayout(layout));
    }
    else
    {
      _executions.at(0)->setInputLayout(onert::ir::IOIndex(index), convertLayout(layout));
    }
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
    if (_execution)
    {
      _execution->setOutputLayout(onert::ir::IOIndex(index), convertLayout(layout));
    }
    else
    {
      _executions.at(_executions.size() - 1)
        ->setOutputLayout(onert::ir::IOIndex(index), convertLayout(layout));
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output_layout : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
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
    auto model = _nnpkg->primary_model();
    auto primary_subgraph = model->primary_subgraph();
    auto ind = primary_subgraph->getInputs().at(index);
    auto &input = primary_subgraph->operands().at(ind);

    // overwrite input shape with the shape from ti
    input.info().shape(new_shape);
  }
  else // when called after nnfw_session::prepare()
  {
    if (_execution)
    {
      _execution->changeInputShape(onert::ir::IOIndex(index), new_shape);
    }
    else
    {
      _executions.at(0)->changeInputShape(onert::ir::IOIndex(index), new_shape);
    }
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
    {
      shape = _execution ? _execution->getInputShape(onert::ir::IOIndex{index})
                         : _executions.at(0)->getInputShape(onert::ir::IOIndex{index});
    }
    auto dtype = primary_subgraph()->operands().at(opidx).typeInfo().type();
    fillTensorInfo(ti, shape, dtype);
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
    {
      shape = _execution
                ? _execution->getOutputShape(onert::ir::IOIndex{index})
                : _executions.at(_executions.size() - 1)->getOutputShape(onert::ir::IOIndex{index});
    }
    auto dtype = primary_subgraph()->operands().at(opidx).typeInfo().type();
    fillTensorInfo(ti, shape, dtype);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

void nnfw_session::make_dependency()
{
  for (uint32_t out_exe = 0; out_exe < _executions.size(); out_exe++)
  {
    auto &out_graph = _executions[out_exe]->primary_subgraph();
    for (uint32_t in_exe = 0; in_exe < _executions.size(); in_exe++)
    {
      if (out_exe == in_exe)
        continue;
      auto &in_graph = _executions[in_exe]->primary_subgraph();
      for (auto out = out_graph._name_to_output_begin(); out != out_graph._name_to_output_end();
           out++)
      {
        auto out_opidx = out_graph.getOutputs().at(out->second);
        auto out_shape = out_graph.operands().at(out_opidx).shape();
        for (auto in = in_graph._name_to_input_begin(); in != in_graph._name_to_input_end(); in++)
        {
          if (out->first != in->first)
            continue;

          auto in_opidx = in_graph.getInputs().at(in->second);
          auto in_shape = in_graph.operands().at(in_opidx).shape();
          if (out_shape.rank() != in_shape.rank())
            continue;

          bool is_same = true;
          for (int32_t i = 0; i < out_shape.rank(); i++)
          {
            if (out_shape.dim(i) != in_shape.dim(i))
            {
              is_same = false;
              break;
            }
          }

          if (is_same)
            _executions[out_exe]->pushNextExe(_executions[in_exe], out->second, in->second);
        }
      }
    }
  }
}

NNFW_STATUS nnfw_session::push_pipeline_input(std::vector<void *> *inputs,
                                              std::vector<uint32_t> *lengths)
{
  static uint32_t count = 0;
  if (inputs->empty())
  {
    _executions[0]->setFinish();
    for (uint32_t i = 0; i < _threads.size(); i++)
    {
      _threads[i].join();
    }
    return NNFW_STATUS_NO_ERROR;
  }
  _executions[0]->asyncIoDescSemWait();
  _executions[0]->createNewAsyncDesc(count++);
  for (uint32_t i = 0; i < inputs->size(); i++)
  {
    _executions[0]->executeAsyncInput(onert::ir::IOIndex(i), inputs->at(i), lengths->at(i));
  }
  _executions[0]->asyncIoDescSemPost();
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::pop_pipeline_output(std::vector<void *> *outputs)
{
  auto results = _executions[_executions.size() - 1]->getAsyncResults();
  while (results->empty())
  {
    if (_executions[_executions.size() - 1]->stopWait())
      return NNFW_STATUS_ERROR;
  }

  auto result = results->front();
  results->pop_front();
  for (uint32_t i = 0; i < result.size(); i++)
    outputs->push_back(result[i]);
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

    auto &options = *_coptions[0];

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

    auto &opcode_to_backend = _coptions[0]->manual_scheduler_options.opcode_to_backend;
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

  auto &options = *_coptions[0];

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

const onert::ir::Graph *nnfw_session::primary_subgraph()
{
  if (_nnpkg != nullptr)
  {
    assert(_execution == nullptr && _executions.empty());
    return _nnpkg->primary_model()->primary_subgraph().get();
  }
  else
  {
    assert(_execution != nullptr || !_executions.empty());
    // TODO Remove const_cast
    // We assumed the graph will not change after compilation, but shape could change
    if (!_executions.empty())
    {
      return &_executions[0]->primary_parentgraph();
    }

    return &_execution->primary_subgraph();
  }
}

NNFW_STATUS nnfw_session::get_config(const char *key, char *value, size_t value_size)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = *_coptions[0];

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
    assert(_nnpkg == nullptr);
    assert(_coptions.empty());
    assert(_execution == nullptr && _executions.empty());
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
    assert(_nnpkg != nullptr);
    assert(!_coptions.empty());
    assert(_execution == nullptr && _executions.empty());
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
    assert(_nnpkg == nullptr);
    assert(!_coptions.empty());
    assert(_execution != nullptr || !_executions.empty());
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
    assert(_nnpkg == nullptr);
    assert(!_coptions.empty());
    assert(_execution != nullptr || !_executions.empty());
    return true;
  }
  return false;
}

bool nnfw_session::isStateFinishedRun()
{
  if (_state == State::FINISHED_RUN)
  {
    assert(_nnpkg == nullptr);
    assert(!_coptions.empty());
    assert(_execution != nullptr || !_executions.empty());
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

NNFW_STATUS nnfw_session::set_backends_per_operation(const char *backend_settings)
{
  if (backend_settings == NULL)
    return NNFW_STATUS_ERROR;

  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  // Backend for all
  auto &ms_options = _coptions[0]->manual_scheduler_options;
  ms_options.setBackendMap(std::string{backend_settings});

  return NNFW_STATUS_NO_ERROR;
}
