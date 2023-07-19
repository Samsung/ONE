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
#include "compiler/CompilerFactory.h"
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

NNFW_STATUS getTensorIndexImpl(const onert::ir::IGraph &graph, const char *tensorname,
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
  try
  {
    auto new_session = std::unique_ptr<nnfw_session>(new nnfw_session());
    new_session->_kernel_registry = std::make_shared<onert::api::CustomKernelRegistry>();
    *session = new_session.release();
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Error during session creation" << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return NNFW_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during session initialization : " << e.what() << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
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
    auto num_models = models.size();
    if (num_models == 0 || (num_models - 1) > onert::ir::ModelIndex::max())
    {
      std::cerr << "Invalid model size - " << std::to_string(num_models) << std::endl;
      return NNFW_STATUS_ERROR;
    }

    for (uint16_t i = 0; i < num_models; ++i)
    {
      auto model_file_path = package_path + std::string("/") + models[i].asString();
      auto model_type = model_types[i].asString();
      auto model = loadModel(model_file_path, model_type);
      if (model == nullptr)
        return NNFW_STATUS_ERROR;
      model->bindKernelBuilder(_kernel_registry->getBuilder());
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

    _nnpkg->verify();
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
    auto compiler = onert::compiler::CompilerFactory::get().create(_nnpkg, _coptions);
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

NNFW_STATUS nnfw_session::prepare_pipeline(const char *)
{
  std::cerr << "Pipeline prepare_pipeline: deprecated feature " << std::endl;
  return NNFW_STATUS_ERROR;
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
    *number = getInputSize();
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
    *number = getOutputSize();
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
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_input_layout : "
              << "run should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

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
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_output_layout : "
              << "run should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

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

    // In this case, if we apply input shape, it will propagate after compilation and excution
    _nnpkg->changeInputShape(index, new_shape);
  }
  else // when called after nnfw_session::prepare()
    _execution->changeInputShape(onert::ir::IOIndex(index), new_shape);

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

    if (index >= getInputSize())
    {
      std::cerr << "Error during nnfw_session::input_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }

    if (isStateModelLoaded())
    {
      auto info = _nnpkg->inputInfo(index);
      fillTensorInfo(ti, info.shape(), info.typeInfo().type());
    }
    else
    {
      auto io_index = onert::ir::IOIndex{index};
      auto shape = _execution->getInputShape(io_index);
      auto dtype = _compiler_artifact->_executors->inputInfo(io_index).typeInfo().type();
      fillTensorInfo(ti, shape, dtype);
    }
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

  try
  {
    if (index >= getOutputSize())
    {
      std::cerr << "Error during nnfw_session::output_tensorinfo, index is out of range."
                << std::endl;
      return NNFW_STATUS_ERROR;
    }

    if (isStateModelLoaded())
    {
      auto info = _nnpkg->outputInfo(index);
      fillTensorInfo(ti, info.shape(), info.typeInfo().type());
    }
    else
    {
      auto io_index = onert::ir::IOIndex{index};
      auto shape = _execution->getOutputShape(io_index);
      auto dtype = _compiler_artifact->_executors->outputInfo(io_index).typeInfo().type();
      fillTensorInfo(ti, shape, dtype);
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::output_tensorinfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::push_pipeline_input(std::vector<void *> *, std::vector<uint32_t> *)
{
  std::cerr << "Pipeline push_pipeline_input: deprecated feature " << std::endl;
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::pop_pipeline_output(std::vector<void *> *)
{
  std::cerr << "Pipeline pop_pipeline_output: deprecated feature " << std::endl;
  return NNFW_STATUS_ERROR;
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
  else
  {
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

const onert::ir::IGraph *nnfw_session::primary_subgraph()
{
  if (_nnpkg != nullptr)
  {
    assert(_execution == nullptr);
    return _nnpkg->primary_model()->primary_subgraph().get();
  }
  else
  {
    assert(_execution != nullptr);
    // We assumed the graph will not change after compilation, but shape could change
    return &_execution->primary_subgraph();
  }
}

uint32_t nnfw_session::getInputSize()
{
  if (isStateInitialized())
    throw std::runtime_error{"Model is not loaded yet"};

  if (isStateModelLoaded())
    return _nnpkg->inputSize();

  // Session is prepared (general inference)
  return _compiler_artifact->_executors->inputSize();
}

uint32_t nnfw_session::getOutputSize()
{
  if (isStateInitialized())
    throw std::runtime_error{"Model is not loaded yet"};

  if (isStateModelLoaded())
    return _nnpkg->outputSize();

  // Session is prepared (general inference)
  return _compiler_artifact->_executors->outputSize();
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
    assert(_execution == nullptr);
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
    assert(_execution == nullptr);
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
    assert(_execution != nullptr);
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
    assert(_execution != nullptr);
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
    assert(_execution != nullptr);
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

#ifdef ONERT_TRAIN
NNFW_STATUS nnfw_session::train_prepare(const nnfw_train_info *info)
{
  // We may need different state to represent training model is loaded
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during model prepare training: ";
    if (_state == State::PREPARED_TRAINING)
      std::cerr << "prepare should be run once";
    else
      std::cerr << "invalid state";
    std::cerr << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    nnfw_train_info tinfo;
    if (info != nullptr)
    {
      tinfo = *info;
    }

    onert::compiler::train::TrainingInfo training_info;
    training_info.setBatchSize(tinfo.batch_size);
    // TODO Set Loss function

    // TODO Set a correct optimizer type and learning_rate
    onert::compiler::train::OptimizerInfo opt_info;
    opt_info.optim_code = onert::exec::train::optimizer::OptimizerCode::SGD;
    opt_info.learning_rate = 0.01;
    training_info.setOptimizerInfo(opt_info);

    auto compiler =
      onert::compiler::CompilerFactory::get().create(_nnpkg, _coptions, &training_info);
    _nnpkg.reset();
    _compiler_artifact = compiler->compile();
    _execution = std::make_unique<onert::exec::Execution>(_compiler_artifact->_executors);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_prepare : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::PREPARED_TRAINING;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_input_tensorinfo : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // Check index is valid: [0, getInputSize())

  // NYI
  (void)index;
  (void)ti;
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::train_expected_tensorinfo(uint32_t index, nnfw_tensorinfo *ti)
{
  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_expected_tensorinfo : invalid state"
              << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // Check index is valid: [0, getExpectedSize())

  // NYI
  (void)index;
  (void)ti;
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::train_set_input(uint32_t index, const void *input,
                                          const nnfw_tensorinfo *input_tensorinfo)
{
  if (input == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_set_input : input buffer is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_set_input : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (index >= getInputSize())
  {
    std::cerr << "Error during nnfw_session::train_set_input : index is out of range" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  if (input_tensorinfo != nullptr)
  {
    std::cerr << "Error during nnfw_session::train_set_input : not supporeted to change tensorinfo"
              << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    auto ind = onert::ir::IOIndex(index);
    auto size = _execution->getInputTotalSize(ind);
    _execution->setInput(ind, input, size);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_set_input : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_set_expected(uint32_t index, const void *expected,
                                             const nnfw_tensorinfo *expected_tensorinfo)
{
  if (expected == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_set_expected : expected buffer is null"
              << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_set_expected : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // Check index is valid: [0, getExpectedSize())

  // Maybe it cannot use general execute->setOutput()
  // because it needs to set training output (maybe loss)
  (void)index;
  (void)expected_tensorinfo;

  // NYI
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::train_run(bool update_weights)
{
  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_run : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    if (update_weights)
    {
      _execution->train(_training_step++);
    }
    else
      _execution->execute();
  }
  catch (const onert::InsufficientBufferSizeException &e)
  {
    // Currently insufficient buffer always means output buffer.
    std::cerr << "Error during nnfw_session::train_run : " << e.what() << std::endl;
    return NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_run : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::FINISHED_TRAINING;
  return NNFW_STATUS_NO_ERROR;
}

float nnfw_session::train_get_loss(uint32_t index)
{
  if (!isStateFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_get_loss : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // Check index is valid: [0, getExpectedSize())

  (void)index;

  // NYI
  throw std::runtime_error{"Return loss: Not implemented yet"};
}

NNFW_STATUS nnfw_session::train_export_circle(const char *path)
{
  if (path == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_export_circle : path is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  // Check training mode is enabled
  if (!isStateFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_export_circle : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // NYI
  return NNFW_STATUS_ERROR;
}

bool nnfw_session::isStatePreparedTraining()
{
  if (_state == State::PREPARED_TRAINING)
  {
    assert(_nnpkg == nullptr);
    assert(!_coptions.empty());
    assert(_execution != nullptr);
    return true;
  }
  else
    return false;
}

bool nnfw_session::isStateFinishedTraining()
{
  if (_state == State::FINISHED_TRAINING)
  {
    assert(_nnpkg == nullptr);
    assert(!_coptions.empty());
    assert(_execution != nullptr);
    return true;
  }
  else
    return false;
}

bool nnfw_session::isStatePreparedOrFinishedTraining()
{
  return isStatePreparedTraining() || isStateFinishedTraining();
}

#endif // ONERT_TRAIN
