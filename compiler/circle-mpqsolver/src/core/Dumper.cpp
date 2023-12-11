/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Dumper.h"

#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include <json.h>
#include <fstream>
#include <sys/stat.h>

using namespace mpqsolver::core;

namespace
{

const std::string default_dtype_key = "default_quantization_dtype";
const std::string default_granularity_key = "default_granularity";
const std::string layers_key = "layers";
const std::string model_key = "model_path";
const std::string layer_name_key = "name";
const std::string layer_dtype_key = "dtype";
const std::string layer_granularity_key = "granularity";

} // namespace

Dumper::Dumper(const std::string &dir_path) : _dir_path(dir_path) {}

void Dumper::setModelPath(const std::string &model_path) { _model_path = model_path; }

void Dumper::dumpMPQConfiguration(const LayerParams &layers, const std::string &def_dtype,
                                  const std::string &def_granularity, const std::string &path) const
{
  Json::Value mpq_data;
  mpq_data[default_dtype_key] = def_dtype;
  mpq_data[default_granularity_key] = def_granularity;
  mpq_data[model_key] = _model_path;

  Json::Value layers_data;
  for (auto &layer : layers)
  {
    Json::Value layer_data;
    layer_data[layer_name_key] = layer->name;
    layer_data[layer_granularity_key] = layer->granularity;
    layer_data[layer_dtype_key] = layer->dtype;
    layers_data.append(layer_data);
  }
  mpq_data[layers_key] = layers_data;

  Json::StreamWriterBuilder builder;
  auto data = Json::writeString(builder, mpq_data);

  writeDataToFile(path, data);
}

void Dumper::prepareDirectory(const std::string &dir_path) const
{
  struct stat sb;
  if (stat(dir_path.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
  {
    if (mkdir(dir_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
    {
      throw std::runtime_error("Failed to create directory for dumping intermediate results");
    }
  }
}

void Dumper::dumpMPQConfiguration(const LayerParams &layers, const std::string &def_dtype,
                                  const std::string &def_granularity, int step) const
{
  prepareDirectory(_dir_path);
  std::string path = _dir_path + "/Configuration_" + std::to_string(step) + ".mpq.json";
  dumpMPQConfiguration(layers, def_dtype, def_granularity, path);
}

void Dumper::dumpFinalMPQ(const LayerParams &layers, const std::string &def_dtype,
                          const std::string &def_granularity) const
{
  prepareDirectory(_dir_path);
  std::string path = _dir_path + "/FinalConfiguration" + ".mpq.json";
  dumpMPQConfiguration(layers, def_dtype, def_granularity, path);
}

void Dumper::writeDataToFile(const std::string &path, const std::string &data) const
{
  std::ofstream file;
  file.open(path);
  file << data;
  file.close();
}

void Dumper::saveCircle(luci::Module *module, std::string &path) const
{
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(module, path);
  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("Failed to export circle model to " + path);
  }
}

void Dumper::dumpQuantized(luci::Module *module, uint32_t step) const
{
  std::string path = _dir_path + "/quantized_" + std::to_string(step) + ".mpq.circle";
  saveCircle(module, path);
}

void Dumper::dumpError(float error, const std::string &tag, const std::string &path) const
{
  std::ofstream file;
  file.open(path, std::ios_base::app);
  file << tag << " " << error << std::endl;
  file.close();
}

void Dumper::prepareForErrorDumping() const
{
  prepareDirectory(_dir_path);
  std::string path = getErrorPath();
  std::ofstream file;
  file.open(path); // create empty
  file.close();
}

void Dumper::dumpQ8Error(float error) const
{
  std::string path = getErrorPath();
  dumpError(error, "Q8", path);
}

void Dumper::dumpQ16Error(float error) const
{
  std::string path = getErrorPath();
  dumpError(error, "Q16", path);
}

void Dumper::dumpMPQError(float error, uint32_t step) const
{
  std::string path = getErrorPath();
  dumpError(error, std::to_string(step), path);
}

void Dumper::dumpMPQError(float error) const
{
  std::string path = getErrorPath();
  dumpError(error, "FINAL", path);
}
