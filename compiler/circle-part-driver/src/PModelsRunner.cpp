/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PModelsRunner.h"

#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>
#include <luci/Importer.h>
#include <luci/Log.h>

#include <foder/FileLoader.h>
#include <crew/PConfig.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

namespace
{

void write_file(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

std::unique_ptr<luci::Module> import_circle(const std::string &filename)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + filename + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());

  return luci::Importer().importModule(circle::GetModel(model_data.data()));
}

void save_shape(const std::string &shape_filename, const luci::CircleOutput *output_node)
{
  if (output_node->rank() == 0)
  {
    write_file(shape_filename, "1", 1);
  }
  else
  {
    auto shape_str = std::to_string(output_node->dim(0).value());
    for (uint32_t j = 1; j < output_node->rank(); j++)
    {
      shape_str += ",";
      shape_str += std::to_string(output_node->dim(j).value());
    }
    write_file(shape_filename, shape_str.c_str(), shape_str.size());
  }
}

} // namespace

namespace prunner
{

bool PModelsRunner::load_config(const std::string &filename)
{
  if (!crew::read_ini(filename, _pconfig))
  {
    std::cerr << "ERROR: Invalid config ini file: '" << filename << "'" << std::endl;
    return false;
  }

  for (auto &part : _pconfig.parts)
  {
    _models_to_run.push_back(part.model_file);
  }
  return true;
}

void PModelsRunner::load_inputs(const std::string &input_prefix, int32_t num_inputs)
{
  LOGGER(l);

  auto its = _pconfig.source.inputs.begin();
  for (int32_t i = 0; i < num_inputs; ++i, ++its)
  {
    std::string filename = input_prefix + std::to_string(i);

    INFO(l) << "Load input data: " << filename << std::endl;
    foder::FileLoader file_loader{filename};

    std::string input_name = *its;
    _data_stage[input_name] = file_loader.load();

    INFO(l) << "Input: [" << input_name << "], size " << _data_stage[input_name].size()
            << std::endl;
  }
}

bool PModelsRunner::run(void)
{
  // TODO add implementation
  return true;
}

void PModelsRunner::save_outputs(const std::string &output_file)
{
  // load source model as we need to get both shape and node name
  // TODO check for unknown shape
  auto source_fname = _pconfig.source.model_file;

  auto module = import_circle(source_fname);

  const auto output_nodes = loco::output_nodes(module->graph());
  for (uint32_t i = 0; i < module->graph()->outputs()->size(); i++)
  {
    const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);

    auto output_name = output_node->name();
    assert(_data_stage.find(output_name) != _data_stage.end());

    auto tensor_data = _data_stage[output_name];
    auto output_filename = output_file + std::to_string(i);

    write_file(output_filename, tensor_data.data(), tensor_data.size());
    save_shape(output_filename + ".shape", output_node);
  }
}

} // namespace prunner
