/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Loader.h"

#include <circle_loader.h>
#include <tflite_loader.h>
#include <util/GeneralConfigSource.h>

#include <json/json.h>

#include <dirent.h>
#include <fstream>

#define MAX_PATH_LENGTH 1024

namespace onert
{
namespace api
{

namespace
{

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

using CfgKeyValues = std::unordered_map<std::string, std::string>;

bool loadConfigure(const std::string cfgfile, CfgKeyValues &keyValues)
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

void setConfigKeyValues(const CfgKeyValues &keyValues)
{
  auto configsrc = std::make_unique<onert::util::GeneralConfigSource>();

  for (auto it = keyValues.begin(); it != keyValues.end(); ++it)
  {
    VERBOSE(NNPKG_CONFIGS) << "(" << it->first << ") = (" << it->second << ")" << std::endl;
    configsrc->set(it->first, it->second);
  }

  onert::util::config_source_ext(std::move(configsrc));
}

} // namespace

Loader::Loader(nnfw_session *session) : _session{session}
{
  // DO NOTHING
}

void Loader::loadCircleBuffer(uint8_t *buffer, size_t size)
{
  try
  {
    _session->_subgraphs = onert::circle_loader::loadModel(buffer, size);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    _session->_subgraphs = nullptr;
  }
}

void Loader::loadModelFile(const std::string &path, const std::string &type)
{
  try
  {
    if (type == "tflite")
    {
      _session->_subgraphs = onert::tflite_loader::loadModel(path);
    }
    else if (type == "circle")
    {
      _session->_subgraphs = onert::circle_loader::loadModel(path);
    }
    else
    {
      std::cerr << "Unsupported model type" << std::endl;
      _session->_subgraphs = nullptr;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    _session->_subgraphs = nullptr;
  }
}

NNFW_STATUS Loader::loadCircleFromBuffer(uint8_t *buffer, size_t size)
{
  if (!_session->isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (!buffer)
    return NNFW_STATUS_UNEXPECTED_NULL;

  if (size == 0)
    return NNFW_STATUS_ERROR;

  loadCircleBuffer(buffer, size);
  if (!_session->_subgraphs)
    return NNFW_STATUS_ERROR;

  _session->_tracing_ctx = std::make_unique<onert::util::TracingCtx>(_session->_subgraphs.get());

  _session->_compiler =
    std::make_unique<onert::compiler::Compiler>(_session->_subgraphs, _session->_tracing_ctx.get());

  _session->_state = nnfw_session::State::MODEL_LOADED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS Loader::loadModelFromModelfile(const char *model_file_path)
{
  if (!_session->isStateInitialized())
    return NNFW_STATUS_INVALID_STATE;

  if (!model_file_path)
  {
    std::cerr << "Model file path is null." << std::endl;
    return NNFW_STATUS_UNEXPECTED_NULL;
  }

  std::string filename{model_file_path};
  if (filename.size() < 8) // .tflite or .circle
  {
    std::cerr << "Invalid model file path." << std::endl;
    return NNFW_STATUS_ERROR;
  }

  std::string model_type = filename.substr(filename.size() - 6, 6);

  loadModelFile(model_file_path, model_type);
  if (!_session->_subgraphs)
    return NNFW_STATUS_ERROR;

  _session->_tracing_ctx = std::make_unique<onert::util::TracingCtx>(_session->_subgraphs.get());

  _session->_compiler =
    std::make_unique<onert::compiler::Compiler>(_session->_subgraphs, _session->_tracing_ctx.get());

  _session->_state = nnfw_session::State::MODEL_LOADED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS Loader::loadNNPackage(const char *package_dir)
{
  if (!_session->isStateInitialized())
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

      CfgKeyValues keyValues;
      if (loadConfigure(filepath, keyValues))
      {
        setConfigKeyValues(keyValues);
      }
    }

    auto model_file_path = package_path + std::string("/") + models[0].asString(); // first model
    auto model_type = model_types[0].asString(); // first model's type

    loadModelFile(model_file_path, model_type);
    if (!_session->_subgraphs)
      return NNFW_STATUS_ERROR;

    _session->_subgraphs->primary()->bindKernelBuilder(_session->_kernel_registry->getBuilder());
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model loading : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _session->_tracing_ctx = std::make_unique<onert::util::TracingCtx>(_session->_subgraphs.get());

  _session->_compiler =
    std::make_unique<onert::compiler::Compiler>(_session->_subgraphs, _session->_tracing_ctx.get());

  _session->_state = nnfw_session::State::MODEL_LOADED;
  return NNFW_STATUS_NO_ERROR;
}

} // namespace api
} // namespace onert
