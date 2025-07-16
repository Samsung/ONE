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

#include "nnfw_session.h"

#include "compiler/CompilerFactory.h"
#include "exporter/CircleExporter.h"
#include "exporter/train/CheckpointExporter.h"
#include "ir/OpCode.h"
#include "json/json.h"
#include "loader/CircleLoader.h"
#include "loader/ModelLoader.h"
#include "loader/TFLiteLoader.h"
#include "loader/TrainInfoLoader.h"
#include "loader/train/CheckpointLoader.h"
#include "util/ConfigSource.h"
#include "util/Exceptions.h"
#include "util/logging.h"
#include "util/TracingCtx.h"

#include <misc/string_helpers.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

std::string trim(std::string_view value)
{
  constexpr std::string_view whitespace = " \t";

  auto begin = value.find_first_not_of(whitespace);
  if (begin == std::string_view::npos)
    return ""; // no content

  auto end = value.find_last_not_of(whitespace);
  return std::string(value.substr(begin, end - begin + 1));
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
  try
  {
    if (model_type == "tflite")
      return onert::loader::loadTFLiteModel(filename.c_str());
    if (model_type == "circle")
      return onert::loader::loadCircleModel(filename.c_str());

    return onert::loader::loadModel(filename, model_type);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Fail to load model: " << e.what() << '\n';
  }

  return std::unique_ptr<onert::ir::Model>(nullptr);
}

std::unique_ptr<onert::ir::train::TrainingInfo>
loadTrainingInfo(const std::shared_ptr<onert::ir::Model> &model)
{
  const auto tinfo_name = onert::loader::TRAININFO_METADATA_NAME;
  if (model->exists_metadata(tinfo_name))
  {
    const auto buffer = model->extract_metadata(tinfo_name);
    return onert::loader::loadTrainingInfo(buffer->base(), buffer->size());
  }
  return std::make_unique<onert::ir::train::TrainingInfo>();
}

uint64_t getBufSize(const nnfw_tensorinfo *info)
{
  static int elmsize[] = {
    sizeof(float),   /* NNFW_TYPE_TENSOR_FLOAT32 = 0 */
    sizeof(int),     /* NNFW_TYPE_TENSOR_INT32 = 1 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_QUANT8_ASYMM = 2 */
    sizeof(bool),    /* NNFW_TYPE_TENSOR_BOOL = 3 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_UINT8 = 4 */
    sizeof(int64_t), /* NNFW_TYPE_TENSOR_INT64 = 5 */
    sizeof(int8_t),  /* NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED = 6 */
    sizeof(int16_t), /* NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED = 7 */
  };

  uint64_t n = 1;
  for (int32_t i = 0; i < info->rank; ++i)
  {
    assert(info->dims[i] >= 0);
    n *= info->dims[i];
  }
  return elmsize[info->dtype] * n;
}
} // namespace

nnfw_session::nnfw_session()
  : _nnpkg{nullptr}, _coptions{onert::compiler::CompilerOptions::fromGlobalConfig()},
    _compiler_artifact{nullptr}, _execution{nullptr}, _kernel_registry{nullptr},
    _train_info{nullptr}, _quant_manager{std::make_unique<onert::odc::QuantizeManager>()},
    _codegen_manager{std::make_unique<onert::odc::CodegenManager>()}, _model_path{}
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
    auto model = onert::loader::loadCircleModel(buffer, size);
    // TODO: Update _model_path if necessary
    _nnpkg = std::make_shared<onert::ir::NNPkg>(std::move(model));
    _train_info = loadTrainingInfo(_nnpkg->primary_model());
    _state = State::MODEL_LOADED;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::load_model_from_path(const char *path)
{
  if (!isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (!path)
  {
    std::cerr << "Path is null." << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!null_terminating(path, MAX_PATH_LENGTH))
  {
    std::cerr << "Path is too long" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    std::filesystem::path filename{path};
    if (!std::filesystem::is_directory(filename) && filename.has_extension())
    {
      std::string model_type = filename.extension().string().substr(1); // + 1 to exclude dot
      return loadModelFile(filename, model_type);
    }

    const auto &package_dir = filename;

    // TODO : add support for zipped package file load
    if (!std::filesystem::is_directory(package_dir))
    {
      std::cerr << "invalid path: " << package_dir << std::endl;
      return NNFW_STATUS_ERROR;
    }

    const auto manifest_file_name = package_dir / "metadata/MANIFEST";
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
      const auto filepath = package_dir / "metadata" / configs[0].asString();

      onert::util::CfgKeyValues keyValues;
      if (loadConfigure(filepath.string(), keyValues))
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

    // Not support backend mapping to operator index for multiple models yet
    // TODO Support this
    if (num_models > 1 && _coptions->manual_scheduler_options.index_to_backend.size() != 0)
    {
      std::cerr << "Cannot set backend to operator index for multiple models" << std::endl;
      return NNFW_STATUS_ERROR;
    }

    for (uint16_t i = 0; i < num_models; ++i)
    {
      const auto model_file_path = package_dir / models[i].asString();
      const auto model_type = model_types[i].asString();
      auto model = loadModel(model_file_path.string(), model_type);
      if (model == nullptr)
        return NNFW_STATUS_ERROR;
      _model_path = model_file_path; // TODO Support multiple models
      model->bindKernelBuilder(_kernel_registry->getBuilder());
      _nnpkg->push(onert::ir::ModelIndex{i}, std::move(model));
    }

    _train_info = loadTrainingInfo(_nnpkg->primary_model());

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
    auto compiler = onert::compiler::CompilerFactory::get().create(_nnpkg, _coptions.get());
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

NNFW_STATUS nnfw_session::set_input(uint32_t index, NNFW_TYPE type, const void *buffer,
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
    // Allow float input internal quantization only
    if (type == NNFW_TYPE_TENSOR_FLOAT32)
      _execution->setInputType(onert::ir::IOIndex(index),
                               onert::ir::TypeInfo(onert::ir::DataType::FLOAT32));
    _execution->setInput(onert::ir::IOIndex(index), buffer, length);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_input : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length)
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
    // Allow float output internal dequantization only
    if (type == NNFW_TYPE_TENSOR_FLOAT32)
      _execution->setOutputType(onert::ir::IOIndex(index),
                                onert::ir::TypeInfo(onert::ir::DataType::FLOAT32));
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

NNFW_STATUS nnfw_session::set_input_type(uint32_t index, NNFW_TYPE type)
{
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during nnfw_session::set_input_type : "
              << "set_input_type should be called before prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    if (type != NNFW_TYPE_TENSOR_FLOAT32)
    {
      std::cerr << "Error during nnfw_session::set_input_type, not supported type" << std::endl;
      return NNFW_STATUS_ERROR;
    }

    _coptions->input_type.insert_or_assign(onert::ir::IOIndex{index},
                                           onert::ir::TypeInfo(onert::ir::DataType::FLOAT32));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_input_type : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  // Not supporting yet
  // TODO Enable this
  std::cerr << "Error during nnfw_session::set_input_type : NYI" << std::endl;
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::set_output_type(uint32_t index, NNFW_TYPE type)
{
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during nnfw_session::set_output_type : "
              << "set_output_type should be called before prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    if (type != NNFW_TYPE_TENSOR_FLOAT32)
    {
      std::cerr << "Error during nnfw_session::set_output_type, not supported type" << std::endl;
      return NNFW_STATUS_ERROR;
    }

    _coptions->output_type.insert_or_assign(onert::ir::IOIndex{index},
                                            onert::ir::TypeInfo(onert::ir::DataType::FLOAT32));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_output_type : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  // Not supporting yet
  // TODO Enable this
  std::cerr << "Error during nnfw_session::set_output_type : NYI" << std::endl;
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti)
{
  // sanity check
  {
    if (isStateInitialized())
    {
      std::cerr << "Error during set_input_tensorinfo : should be run after load_model"
                << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    if (ti == nullptr)
    {
      std::cerr << "Error during nnfw_session::set_input_tensorinfo : tensorinfo is null"
                << std::endl;
      return NNFW_STATUS_UNEXPECTED_NULL;
    }

    if (ti->rank <= 0 || ti->rank > NNFW_MAX_RANK)
    {
      std::cerr << "unsupported rank: " << ti->rank << std::endl;
      return NNFW_STATUS_ERROR;
    }

    for (int32_t i = 0; i < ti->rank; ++i)
    {
      if (ti->dims[i] <= 0)
      {
        std::cerr << "dim must be positive integer but was " << ti->dims[i] << std::endl;
        return NNFW_STATUS_ERROR;
      }
    }
  }

  onert::ir::Shape new_shape(ti->rank);
  for (int32_t i = 0; i < ti->rank; i++)
    new_shape.dim(i) = ti->dims[i];

  if (!isStatePreparedOrFinishedRun())
  {

    // In this case, if we apply input shape, it will propagate after compilation and excution
    _nnpkg->changeInputShape(index, new_shape);
  }
  else // when called after nnfw_session::prepare()
    _execution->changeInputShape(onert::ir::IOIndex(index), new_shape);

  return NNFW_STATUS_NO_ERROR;
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

NNFW_STATUS nnfw_session::register_custom_operation(const std::string &id,
                                                    nnfw_custom_eval eval_func)
{
  _kernel_registry->registerKernel(id, eval_func);
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::get_output(uint32_t index, nnfw_tensorinfo *ti, const void **out_buffer)
{
  if (ti == nullptr)
  {
    std::cerr << "Error during nnfw_session::get_output : tensor info is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (out_buffer == nullptr)
  {
    std::cerr << "Error during nnfw_session::get_output : output buffer is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!isStateFinishedRun())
  {
    std::cerr << "Error during nnfw_session::get_output : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    if (index >= getOutputSize())
    {
      std::cerr << "Error during nnfw_session::get_output, index " << index
                << " is out of range. (output count: " << getOutputSize() << ")" << std::endl;
      return NNFW_STATUS_ERROR;
    }

    if (!_coptions->internal_output_alloc)
    {
      std::cerr << "Error during nnfw_session::get_output: "
                << "internal output allocation is not enabled. "
                << "Call nnfw_set_prepare_config(session, "
                   "NNFW_PREPARE_CONFIG_ENABLE_INTERNAL_OUTPUT_ALLOC, \"true\") "
                << "before nnfw_prepare()." << std::endl;
      return NNFW_STATUS_ERROR;
    }

    auto io_index = onert::ir::IOIndex{index};
    const auto &info = _compiler_artifact->_executors->outputInfo(io_index);
    const auto &shape = info.shape();
    const auto &dtype = info.typeInfo().type();
    fillTensorInfo(ti, shape, dtype);

    *out_buffer = _compiler_artifact->_executors->outputBuffer(io_index);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::get_output : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
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

    using namespace onert::util;

    _coptions->backend_list = nnfw::misc::split(std::string{backends}, ';');
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_available_backends : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_workspace(const char *dir)
{
  // TODO Check dir read & write permission

  if (!dir)
    return NNFW_STATUS_UNEXPECTED_NULL;

  if (!isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  _coptions->workspace_dir = std::string(dir);

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::deprecated(const char *msg)
{
  std::cerr << msg << std::endl;
  return NNFW_STATUS_DEPRECATED_API;
}

NNFW_STATUS nnfw_session::set_config(const char *key, const char *value)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  using namespace onert::util;

  const std::string skey = key;

  if (skey == config::GRAPH_DOT_DUMP)
  {
    _coptions->graph_dump_level = toInt(value);
  }
  else if (skey == config::EXECUTOR)
  {
    _coptions->executor = value;
  }
  else if (skey == config::OP_BACKEND_ALLOPS)
  {
    _coptions->manual_scheduler_options.backend_for_all = value;
  }
  else if (skey == config::USE_SCHEDULER)
  {
    _coptions->he_scheduler = toBool(value);
  }
  else if (skey == config::PROFILING_MODE)
  {
    _coptions->he_profiling_mode = toBool(value);
  }
  else if (skey == config::ONERT_LOG_ENABLE || skey == config::NUM_THREADS)
  {
    onert::util::CfgKeyValues keyValues;
    keyValues[skey] = std::string(value);
    onert::util::setConfigKeyValues(keyValues);

    if (skey == config::ONERT_LOG_ENABLE)
    {
      UPDATE_VERBOSE_CONFIG();
    }
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

NNFW_STATUS nnfw_session::loadModelFile(const std::string &model_file_path,
                                        const std::string &model_type)
{
  auto model = loadModel(model_file_path, model_type);
  if (model == nullptr)
    return NNFW_STATUS_ERROR;

  _nnpkg = std::make_shared<onert::ir::NNPkg>(std::move(model));
  _model_path = std::filesystem::path(model_file_path);
  _compiler_artifact.reset();
  _execution.reset();
  _train_info = loadTrainingInfo(_nnpkg->primary_model());
  _state = State::MODEL_LOADED;

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::get_config(const char *key, char *value, size_t value_size)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

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
    if (_coptions->backend_list.size() == 0)
      return NNFW_STATUS_NO_ERROR; // no setting backend is not an error of get_config_str()

    auto str =
      nnfw::misc::join(_coptions->backend_list.begin(), _coptions->backend_list.end(), ";");

    if (!check_boundary(value_size, str))
      return NNFW_STATUS_ERROR;

    strncpy(value, str.c_str(), value_size);
  }
  else if (skey == onert::util::config::EXECUTOR)
  {
    if (!check_boundary(value_size, _coptions->executor))
      return NNFW_STATUS_ERROR;

    strncpy(value, _coptions->executor.c_str(), _coptions->executor.length());
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

  // Not supported multiple model
  // TODO Support this
  if (_nnpkg->model_count() > 1)
  {
    std::cerr << "Not supported multiple model" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    // Backend for all
    auto &ms_options = _coptions->manual_scheduler_options;
    ms_options.setBackendMap(std::string{backend_settings});
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_backends_per_operation" << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_get_traininfo(nnfw_train_info *info)
{
  if (isStateInitialized())
  {
    // There is no _train_info in INITIALIZED, since _train_info is set when a model loaded
    std::cerr << "Error during nnfw_session::train_get_traininfo : invalid state";
    return NNFW_STATUS_INVALID_STATE;
  }

  if (info == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_get_traininfo : info is nullptr" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  // after model loaded, it ensures that _train_info is not nullptr
  assert(_train_info != nullptr);

  auto convertLossCode = [](const onert::ir::train::LossCode &code) -> NNFW_TRAIN_LOSS {
    switch (code)
    {
      case onert::ir::train::LossCode::Undefined:
        return NNFW_TRAIN_LOSS_UNDEFINED;
      case onert::ir::train::LossCode::MeanSquaredError:
        return NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR;
      case onert::ir::train::LossCode::CategoricalCrossentropy:
        return NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY;
      default:
        throw std::runtime_error{"fail to convert ir::train::LossCode"};
    }
  };

  auto convertLossReduction =
    [](const onert::ir::train::LossReductionType &type) -> NNFW_TRAIN_LOSS_REDUCTION {
    switch (type)
    {
      case onert::ir::train::LossReductionType::Undefined:
        return NNFW_TRAIN_LOSS_REDUCTION_UNDEFINED;
      case onert::ir::train::LossReductionType::SumOverBatchSize:
        return NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE;
      case onert::ir::train::LossReductionType::Sum:
        return NNFW_TRAIN_LOSS_REDUCTION_SUM;
      default:
        throw std::runtime_error{"fail to convert from ir::train::LossReductionType"};
        break;
    }
  };

  auto convertOptimizerCode =
    [](const onert::ir::train::OptimizerCode &code) -> NNFW_TRAIN_OPTIMIZER {
    switch (code)
    {
      case onert::ir::train::OptimizerCode::Undefined:
        return NNFW_TRAIN_OPTIMIZER_UNDEFINED;
      case onert::ir::train::OptimizerCode::SGD:
        return NNFW_TRAIN_OPTIMIZER_SGD;
      case onert::ir::train::OptimizerCode::Adam:
        return NNFW_TRAIN_OPTIMIZER_ADAM;
      default:
        throw std::runtime_error{"fail to convert from ir::train::OptimizerCode"};
    }
  };

  const auto &loss = _train_info->lossInfo();
  const auto &optim = _train_info->optimizerInfo();

  try
  {
    info->learning_rate = optim.learning_rate;
    info->batch_size = _train_info->batchSize();
    info->loss_info.loss = convertLossCode(loss.loss_code);
    info->loss_info.reduction_type = convertLossReduction(loss.reduction_type);
    info->opt = convertOptimizerCode(optim.optim_code);

    if (_train_info->getTrainableOps().size() > 0)
    {
      const uint32_t first_trainable_idx = _train_info->getTrainableOps().cbegin()->value();
      const uint32_t last_trainable_idx = _train_info->getTrainableOps().crbegin()->value();
      const uint32_t ops_size = primary_subgraph()->operations().size();
      const uint32_t trainable_indexes_range = last_trainable_idx - first_trainable_idx + 1;

      // check if trainable ops set contains continuous indexes on the back of the set
      if (last_trainable_idx == ops_size - 1 &&
          trainable_indexes_range == _train_info->getTrainableOps().size())
      {
        // check if all ops are trainable
        if (0 == first_trainable_idx)
        {
          info->num_of_trainable_ops = NNFW_TRAIN_TRAINABLE_ALL;
        }
        else
        {
          info->num_of_trainable_ops = trainable_indexes_range;
        }
      }
      else
      {
        info->num_of_trainable_ops = NNFW_TRAIN_TRAINABLE_INCORRECT_STATE;
        std::cerr << "conversion from set of trainable ops to num_of_trainable_ops is impossible"
                  << std::endl;
        return NNFW_STATUS_INVALID_STATE;
      }
    }
    else
    {
      // no layer will be trained
      info->num_of_trainable_ops = NNFW_TRAIN_TRAINABLE_NONE;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_get_traininfo" << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_set_traininfo(const nnfw_train_info *info)
{
  if (not isStateModelLoaded())
  {
    std::cerr << "Error during nnfw_session::train_set_traininfo : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (info == nullptr)
  {
    std::cerr << "nnfw_session::train_set_traininfo : info is nullptr" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  // after model loaded, it ensures that _train_info is not nullptr
  assert(_train_info != nullptr);

  auto convertLossType = [](const int &type) {
    if (type == NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR)
      return onert::ir::train::LossCode::MeanSquaredError;
    else if (type == NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY)
      return onert::ir::train::LossCode::CategoricalCrossentropy;
    else
      throw std::runtime_error("not supported loss type");
  };

  auto convertLossReductionType = [](const int &type) {
    if (type == NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE)
      return onert::ir::train::LossReductionType::SumOverBatchSize;
    else if (type == NNFW_TRAIN_LOSS_REDUCTION_SUM)
      return onert::ir::train::LossReductionType::Sum;
    else
      throw std::runtime_error("not supported loss reduction type");
  };

  auto convertOptType = [](const int &type) {
    if (type == NNFW_TRAIN_OPTIMIZER_SGD)
      return onert::ir::train::OptimizerCode::SGD;
    else if (type == NNFW_TRAIN_OPTIMIZER_ADAM)
      return onert::ir::train::OptimizerCode::Adam;
    else
      throw std::runtime_error("not supported optimizer type");
  };

  try
  {
    onert::ir::train::LossInfo loss_info;
    loss_info.loss_code = convertLossType(info->loss_info.loss);
    loss_info.reduction_type = convertLossReductionType(info->loss_info.reduction_type);

    onert::ir::train::OptimizerInfo opt_info;
    opt_info.learning_rate = info->learning_rate;
    opt_info.optim_code = convertOptType(info->opt);

    _train_info->setBatchSize(info->batch_size);
    _train_info->setLossInfo(loss_info);
    _train_info->setOptimizerInfo(opt_info);

    if (info->num_of_trainable_ops < -1)
    {
      std::cerr << "Error during nnfw_session::train_set_traininfo: provided num_of_trainable_ops "
                   "has incorrect value: "
                << info->num_of_trainable_ops << std::endl;
      return NNFW_STATUS_ERROR;
    }

    const uint32_t ops_size = primary_subgraph()->operations().size();
    std::set<onert::ir::OperationIndex> trainable_ops;

    if (NNFW_TRAIN_TRAINABLE_ALL == info->num_of_trainable_ops)
    {
      for (uint32_t idx = 0; idx < ops_size; ++idx)
      {
        trainable_ops.emplace(idx);
      }
    }
    else
    {
      if (static_cast<uint32_t>(info->num_of_trainable_ops) > ops_size)
      {
        std::cerr
          << "Error during nnfw_session::train_set_traininfo: provided num_of_trainable_ops="
          << info->num_of_trainable_ops << " is out of operators range equals: " << ops_size
          << std::endl;
        return NNFW_STATUS_ERROR;
      }
      for (uint32_t i = 1; i <= static_cast<uint32_t>(info->num_of_trainable_ops); ++i)
      {
        trainable_ops.emplace(ops_size - i);
      }
    }
    // Note that possible setting an empty trainable_ops set (for NNFW_TRAIN_TRAINABLE_NONE value)
    _train_info->setTrainableOps(trainable_ops);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_set_traininfo : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_prepare()
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

  // after model loaded, it ensures that _train_info is not nullptr
  assert(_train_info != nullptr);

  try
  {
    if (not _train_info->isValid())
      throw std::runtime_error{"training info is not valid"};

    // initialize trainingStep count
    _train_info->trainingStep() = 0;

    auto compiler =
      onert::compiler::CompilerFactory::get().create(_nnpkg, _coptions.get(), _train_info.get());
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

  try
  {
    auto ind = onert::ir::IOIndex(index);
    auto size = _execution->getInputTotalSize(ind);
    if (input_tensorinfo && getBufSize(input_tensorinfo) != size)
    {
      std::cerr
        << "Error during nnfw_session::train_set_input : not supporeted to change tensorinfo"
        << std::endl;
      return NNFW_STATUS_ERROR;
    }

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

  if (index >= getOutputSize())
  {
    std::cerr << "Error during nnfw_session::train_set_expected : index is out of range"
              << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    auto output_ind = onert::ir::IOIndex(index);
    auto size = _execution->getOutputTotalSize(output_ind);
    if (expected_tensorinfo && getBufSize(expected_tensorinfo) != size)
    {
      std::cerr << "Error during nnfw_session::train_set_expected : invalid tensorinfo"
                << std::endl;
      return NNFW_STATUS_ERROR;
    }

    // NOTE Find the loss input index
    // Input is added as many as the number of outputs.
    // The loss index is calculated from the value obtained by subtracting the
    // total output(added loss input) from the total input size.
    auto input_index = getInputSize() - getOutputSize() + index;
    auto input_ind = onert::ir::IOIndex(input_index);
    _execution->setInput(input_ind, expected, size);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_set_expected : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_set_output(uint32_t index, NNFW_TYPE /*type*/, void *buffer,
                                           size_t length)
{
  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_set_output : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (!buffer && length != 0)
  {
    std::cerr << "Error during nnfw_session::train_set_output : given buffer is NULL but the "
                 "length is not 0"
              << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    _execution->setOutput(onert::ir::IOIndex(index), buffer, length);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_set_output : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
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
      auto &training_step = _train_info->trainingStep();
      _execution->train(training_step++);
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

NNFW_STATUS nnfw_session::train_get_loss(uint32_t index, float *loss)
{
  if (loss == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_get_loss : loss is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!isStateFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_get_loss : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (index >= getOutputSize())
  {
    std::cerr << "Error during nnfw_session::train_get_loss : index is out of range" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  try
  {
    auto ind = onert::ir::IOIndex(index);
    *loss = _execution->getLoss(ind);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_get_loss : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
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

  try
  {
    onert::exporter::CircleExporter exporter(_model_path.string(), std::string{path});
    exporter.updateWeight(_execution);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_export_circle : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_export_circleplus(const char *path)
{
  if (path == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_export_circleplus : path is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_export_circleplus : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    onert::exporter::CircleExporter exporter(_model_path.string(), std::string{path});
    exporter.updateWeight(_execution);
    exporter.updateMetadata(_train_info);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_export_circleplus : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_import_checkpoint(const char *path)
{
  if (path == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_import_checkpoint : path is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  if (!isStatePreparedOrFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_import_checkpoint : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    onert::loader::train::loadCheckpoint(path, _train_info, _execution);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_import_checkpoint : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::train_export_checkpoint(const char *path)
{
  if (path == nullptr)
  {
    std::cerr << "Error during nnfw_session::train_export_checkpoint : path is null" << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  // Check training mode is enabled
  if (!isStateFinishedTraining())
  {
    std::cerr << "Error during nnfw_session::train_export_checkpoint : invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    onert::exporter::train::exportCheckpoint(path, _train_info, _execution);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::train_export_checkpoint : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

bool nnfw_session::isStatePreparedTraining()
{
  if (_state == State::PREPARED_TRAINING)
  {
    assert(_nnpkg == nullptr);
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

NNFW_STATUS nnfw_session::set_quantization_type(NNFW_QUANTIZE_TYPE qtype)
{
  using onert::odc::QuantizeType;
  try
  {
    if (isStateInitialized() || isStateRunning())
    {
      std::cerr << "invalid state" << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    QuantizeType odc_qtype = onert::odc::ODC_QTYPE_NOT_SET;
    switch (qtype)
    {
      case NNFW_QUANTIZE_TYPE_U8_ASYM:
        odc_qtype = onert::odc::ODC_QTYPE_U8_ASYM;
        break;
      case NNFW_QUANTIZE_TYPE_I16_SYM:
        odc_qtype = onert::odc::ODC_QTYPE_I16_SYM;
        break;
      case NNFW_QUANTIZE_TYPE_WO_I8_SYM:
        odc_qtype = onert::odc::ODC_QTYPE_WO_I8_SYM;
        break;
      case NNFW_QUANTIZE_TYPE_WO_I16_SYM:
        odc_qtype = onert::odc::ODC_QTYPE_WO_I16_SYM;
        break;
      default:
        return NNFW_STATUS_INVALID_STATE;
    }
    _quant_manager->quantizeType(odc_qtype);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_quantization_type : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_quantized_model_path(const char *path)
{
  try
  {
    if (isStateInitialized() || isStateRunning())
    {
      std::cerr << "invalid state" << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    _quant_manager->exportModelPath(std::string(path));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_quantized_model_path : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::quantize()
{
  try
  {
    if (isStateInitialized() || isStateRunning())
    {
      std::cerr << "invalid state" << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    auto result = _quant_manager->quantize(_model_path.string());
    if (!result)
      return NNFW_STATUS_INVALID_STATE;

    // Replace model
    // TODO Support buffer replace, not file reload
    return loadModelFile(_quant_manager->exportModelPath(), "circle");
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::quantize : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
}

NNFW_STATUS nnfw_session::set_codegen_model_path(const char *path)
{
  try
  {
    if (isStateInitialized() || isStateRunning())
    {
      std::cerr << "invalid state" << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    assert(_codegen_manager != nullptr);
    _codegen_manager->exportModelPath(std::string(path));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_codegen_model_path : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::codegen(const char *target, NNFW_CODEGEN_PREF pref)
{
  try
  {
    if (isStateInitialized() || isStateRunning())
    {
      std::cerr << "Error during nnfw_session::codegen : Invalid state" << std::endl;
      return NNFW_STATUS_INVALID_STATE;
    }

    std::string target_str{target};
    if (target_str.empty() || target_str.size() < 5 ||
        target_str.substr(target_str.size() - 4) != "-gen")
    {
      std::cerr << "Error during nnfw_session::codegen : Invalid target" << std::endl;
      return NNFW_STATUS_ERROR;
    }

    onert::odc::CodegenPreference codegen_pref;
    switch (pref)
    {
      case NNFW_CODEGEN_PREF_DEFAULT:
        codegen_pref = onert::odc::CodegenPreference::CODEGEN_PREF_DEFAULT;
        break;
      case NNFW_CODEGEN_PREF_PERFORMANCE_FIRST:
        codegen_pref = onert::odc::CodegenPreference::CODEGEN_PREF_PERFORMANCE_FIRST;
        break;
      case NNFW_CODEGEN_PREF_MEMORY_FIRST:
        codegen_pref = onert::odc::CodegenPreference::CODEGEN_PREF_MEMORY_FIRST;
        break;
      case NNFW_CODEGEN_PREF_COMPILE_TIME_FIRST:
        codegen_pref = onert::odc::CodegenPreference::CODEGEN_PREF_COMPILE_TIME_FIRST;
        break;
      default:
        std::cerr << "Error during nnfw_session::codegen : Invalid preference" << std::endl;
        return NNFW_STATUS_ERROR;
    }

    assert(_codegen_manager != nullptr);
    auto export_model_path = std::filesystem::path(_codegen_manager->exportModelPath());
    const auto model_type = target_str.substr(0, target_str.size() - 4);
    // If the export_model_path is not set, it generates a compiled model path
    // automatically.
    if (export_model_path.empty())
    {
      // The compiled model path is the same directory of the original model/package with
      // target backend extension.
      export_model_path = _model_path.replace_extension(model_type);
      _codegen_manager->exportModelPath(export_model_path.string());
    }

    _codegen_manager->codegen(_model_path, target, codegen_pref);

    // Replace model
    // TODO Support buffer replace, not file reload
    return loadModelFile(export_model_path, model_type);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::compile : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
}

NNFW_STATUS nnfw_session::set_prepare_config(const NNFW_PREPARE_CONFIG key, const char *)
{
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during nnfw_session::set_prepare_config : Invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  switch (key)
  {
    case NNFW_PREPARE_CONFIG_PROFILE:
      _coptions->he_profiling_mode = true;
      break;
    case NNFW_ENABLE_INTERNAL_OUTPUT_ALLOC:
      _coptions->internal_output_alloc = true;
      break;
    default:
      return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::reset_prepare_config()
{
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during nnfw_session::reset_prepare_config : Invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  _coptions->he_profiling_mode = false;

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_execute_config(const NNFW_RUN_CONFIG key, const char *)
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_execution_config : Invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  switch (key)
  {
    case NNFW_RUN_CONFIG_DUMP_MINMAX:
      if (_coptions->workspace_dir.empty())
        return NNFW_STATUS_ERROR;
      _execution->executionOptions().dump_minmax = true;
      break;
    case NNFW_RUN_CONFIG_TRACE:
      if (_coptions->workspace_dir.empty())
        return NNFW_STATUS_ERROR;
      _execution->executionOptions().trace = true;
      break;
    case NNFW_RUN_CONFIG_PROFILE:
      _execution->executionOptions().profile = true;
      break;
    default:
      return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::reset_execute_config()
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::set_execution_config : Invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  _execution->executionOptions().dump_minmax = false;
  _execution->executionOptions().trace = false;
  _execution->executionOptions().profile = false;

  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_odc_param_minmax_records_count(int minmax_records_count)
{
  if (isStateInitialized() || isStateRunning())
  {
    std::cerr << "invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (_quant_manager->setMinMaxRecordsThreshold(minmax_records_count))
    return NNFW_STATUS_NO_ERROR;
  else
    return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_session::delete_odc_minmax_file()
{
  if (isStateRunning())
  {
    std::cerr << "invalid state" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  if (_quant_manager->deleteMinMaxFile())
    return NNFW_STATUS_NO_ERROR;
  else
    return NNFW_STATUS_ERROR;
}

// run with auto compilation
NNFW_STATUS nnfw_session::run_with_auto_compilation(const char *target, NNFW_CODEGEN_PREF pref)
{

  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run_with_auto_compilation : "
              << "run should be after preparation" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // Check quantization and code-generation parameters
  std::string target_str{target};
  if (_quant_manager->exportModelPath().empty() || _codegen_manager->exportModelPath().empty() ||
      target_str.empty() || target_str.substr(target_str.size() - 4) != "-gen")
  {
    std::cerr << "Error during nnfw_session::run_with_auto_compilation : "
              << "quantization and code generation parameters should be set" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  // Odc: auto compilation with hidden switching mechanizm
  // Check is model already quantized or compiled
  std::ifstream file_quantized_model(_quant_manager->exportModelPath());
  std::ifstream file_compiled_model(_codegen_manager->exportModelPath());

  if (!file_quantized_model.good() && !file_compiled_model.good())
  {
    // Run float model and try to quantize it
    {
      // Save execution options
      auto saved_options = _execution->executionOptions();
      // turn on minmax recording
      _execution->executionOptions().dump_minmax = true;

      try
      {
        _execution->execute();
      }
      catch (const onert::InsufficientBufferSizeException &e)
      {
        // Currently insufficient buffer always means output buffer.
        std::cerr << "Error during nnfw_session::run_with_auto_compilation : " << e.what()
                  << std::endl;
        return NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE;
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error during nnfw_session::run_with_auto_compilation : " << e.what()
                  << std::endl;
        return NNFW_STATUS_ERROR;
      }

      _state = State::FINISHED_RUN;

      // restore min_max option to user defined state
      _execution->executionOptions().dump_minmax = saved_options.dump_minmax;

      // if enough statistics are collected, then run the quantization
      if (_quant_manager->readyForQuantize())
      {
        try
        {
          if (isStateInitialized() || isStateRunning())
          {
            std::cerr << "invalid state" << std::endl;
            return NNFW_STATUS_INVALID_STATE;
          }

          auto result = _quant_manager->quantize(_model_path);
          if (!result)
            return NNFW_STATUS_INVALID_STATE;

          // remove minmax file
          result = _quant_manager->deleteMinMaxFile();
          if (!result)
            return NNFW_STATUS_INVALID_STATE;
        }
        catch (const std::exception &e)
        {
          std::cerr
            << "Error during nnfw_session::run_with_auto_compilation in quantize operation: "
            << e.what() << std::endl;
          return NNFW_STATUS_ERROR;
        }
      }
    }
  }
  else
  {
    // run compiled or quantized model
    NNFW_STATUS status;

    // turn off minmax recording
    _execution->executionOptions().dump_minmax = false;

    // save initial buffers if quantized model or compiled model is not loaded
    if (_autoCompilationState == nnfw_session::AutoCompilationState::INITIAL_STATE)
    {
      auto dotidx = _codegen_manager->exportModelPath().rfind('.');
      if (dotidx == std::string::npos)
      {
        std::cerr << "Error during nnfw_session::run_with_auto_compilation : Invalid compiled "
                     "model path. Please use a "
                     "path that includes the extension."
                  << std::endl;
        return NNFW_STATUS_ERROR;
      }

      std::string compiled_model_type =
        _codegen_manager->exportModelPath().substr(dotidx + 1); // + 1 to exclude dot

      dotidx = _quant_manager->exportModelPath().rfind('.');
      if (dotidx == std::string::npos)
      {
        std::cerr << "Error during nnfw_session::run_with_auto_compilation : Invalid quantized "
                     "model path. Please use a "
                     "path that includes the extension."
                  << std::endl;
        return NNFW_STATUS_ERROR;
      }
      std::string quantized_model_type =
        _quant_manager->exportModelPath().substr(dotidx + 1); // + 1 to exclude dot

      // Save initial (float) input and output buffers
      auto input_size = _compiler_artifact->_executors->inputSize();
      auto output_size = _compiler_artifact->_executors->outputSize();

      std::vector<const void *> _input_buffers;
      std::vector<void *> _output_buffers;

      // Save Inputs buffers
      for (size_t input_index = 0; input_index < input_size; input_index++)
      {
        auto io_input_index = onert::ir::IOIndex(input_index);
        auto input_Shape = _execution->getInputShape(io_input_index);
        auto input_buffer = _execution->getInputBuffer(io_input_index);

        _input_buffers.push_back(input_buffer);
      }

      // Save Outputs buffers
      for (size_t output_index = 0; output_index < output_size; output_index++)
      {
        auto io_output_index = onert::ir::IOIndex(output_index);

        auto output_Shape = _execution->getOutputShape(io_output_index);
        auto output_buffer = _execution->getOutputBuffer(io_output_index);

        _output_buffers.push_back(output_buffer);
      }

      // Save execution options
      auto saved_options = _execution->executionOptions();

      // if there is compiled model - try to load it
      if (file_compiled_model.good())
      {
        // load compiled model
        status = loadModelFile(_codegen_manager->exportModelPath(), compiled_model_type);
        if (status == NNFW_STATUS_NO_ERROR)
        {
          _autoCompilationState = nnfw_session::AutoCompilationState::COMPILED_MODEL_LOADED;
        }
      }
      else // there is no compiled model - try to compile and load it
      {

        // avoiding code duplication use existing "codegen" function. Set up _model_path for the
        // codegen function.
        // TODO: change it if codegen function will be generalized
        _model_path = _quant_manager->exportModelPath();

        // try to compile and load compiled model
        status = codegen(target, pref);
        if (status == NNFW_STATUS_NO_ERROR)
        {
          _autoCompilationState = nnfw_session::AutoCompilationState::COMPILED_MODEL_LOADED;
          // TODO delete quantized model
        }
      }

      // loading compiled model is fail - try to load quantized model
      if (_autoCompilationState != nnfw_session::AutoCompilationState::COMPILED_MODEL_LOADED)
      {
        // load quantized model
        status = loadModelFile(_quant_manager->exportModelPath(), quantized_model_type);
        if (status != NNFW_STATUS_NO_ERROR)
          return status;
        else
          _autoCompilationState = nnfw_session::AutoCompilationState::QUANTIZED_MODEL_LOADED;
      }

      status = prepare();
      if (status != NNFW_STATUS_NO_ERROR)
        return status;

      // Restore execution options
      _execution->executionOptions() = saved_options;

      // Restore inputs to the quantized or compiled model
      for (uint32_t input_index = 0; input_index < _input_buffers.size(); input_index++)
      {
        nnfw_tensorinfo ti;
        status = input_tensorinfo(input_index, &ti);
        if (status != NNFW_STATUS_NO_ERROR)
          return status;

        ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
        auto input_size_in_bytes = getBufSize(&ti);

        status = set_input(input_index, ti.dtype, _input_buffers[input_index], input_size_in_bytes);

        if (status != NNFW_STATUS_NO_ERROR)
          return status;
      }

      // Restore outputs to the quantized or compiled model
      for (uint32_t output_index = 0; output_index < _output_buffers.size(); output_index++)
      {

        nnfw_tensorinfo ti;
        status = output_tensorinfo(output_index, &ti);
        if (status != NNFW_STATUS_NO_ERROR)
          return status;

        ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;

        uint64_t output_size_in_bytes = getBufSize(&ti);

        status =
          set_output(output_index, ti.dtype, _output_buffers[output_index], output_size_in_bytes);
        if (status != NNFW_STATUS_NO_ERROR)
          return status;
      }
    }

    // Run quantized model
    if (!isStatePreparedOrFinishedRun())
    {
      std::cerr << "Error during nnfw_session::run_with_auto_compilation : "
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
      std::cerr << "Error during nnfw_session::run_with_auto_compilation : " << e.what()
                << std::endl;
      return NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE;
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during nnfw_session::run_with_auto_compilation : " << e.what()
                << std::endl;
      return NNFW_STATUS_ERROR;
    }

    _state = State::FINISHED_RUN;
  }

  return NNFW_STATUS_NO_ERROR;
}
