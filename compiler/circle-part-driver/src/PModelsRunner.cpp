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
#include <luci/IR/DataTypeHelper.h>
#include <luci/Importer.h>
#include <luci/Log.h>
#include <luci_interpreter/Interpreter.h>

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

template <typename NodeT> size_t tensor_size(const NodeT *node)
{
  uint32_t tsize = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    assert(node->dim(i).known());
    tsize *= node->dim(i).value();
  }
  return tsize;
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

/**
 * @brief return true if all inputs of the model is ready in _data_storage
 */
bool PModelsRunner::is_input_ready(const RunModel &model)
{
  for (auto &part : _pconfig.parts)
  {
    if (part.model_file != model)
      continue;

    for (auto &input : part.inputs)
    {
      auto it = _data_stage.find(input);
      if (it == _data_stage.end())
        return false;
    }
  }
  return true;
}

bool PModelsRunner::run(void)
{
  LOGGER(l);

  // for each partitioned model, if the inputs of the model are ready, run the model
  do
  {
    bool found_model = false;

    for (auto it = _models_to_run.begin(); it != _models_to_run.end(); ++it)
    {
      auto model_fname = *it;

      INFO(l) << "Check model input ready: " << model_fname << std::endl;
      if (is_input_ready(model_fname))
      {
        found_model = true;

        INFO(l) << "Run model: " << model_fname << std::endl;
        auto module = import_circle(model_fname);

        luci_interpreter::Interpreter interpreter(module.get());

        // Set input
        const auto input_nodes = loco::input_nodes(module->graph());
        int32_t num_inputs = static_cast<int32_t>(input_nodes.size());
        for (int32_t i = 0; i < num_inputs; i++)
        {
          const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[i]);

          auto input_name = input_node->name();
          assert(_data_stage.find(input_name) != _data_stage.end());

          auto input_data = _data_stage[input_name];

          interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());
        }

        // Run interpreter
        interpreter.interpret();
        INFO(l) << "Run model: " << model_fname << " done" << std::endl;

        // Get output.
        const auto output_nodes = loco::output_nodes(module->graph());
        for (uint32_t i = 0; i < module->graph()->outputs()->size(); i++)
        {
          const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);
          auto output_name = output_node->name();

          Buffer output_data(tensor_size(output_node));

          interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());

          // There should not exist same output names
          // TODO check with multiple virtual outputs
          assert(_data_stage.find(output_name) == _data_stage.end());
          _data_stage[output_name] = output_data;
        }

        // We've ran this model, remove from the model list
        _models_to_run.erase(it);
        break;
      }
    }

    if (not found_model)
    {
      std::cerr << "ERROR: model partition or configuration has problems" << std::endl;
      return false;
    }
  } while (not _models_to_run.empty());

  return true;
}

void PModelsRunner::save_outputs(const std::string &output_file)
{
  LOGGER(l);

  // load source model as we need to get both shape and node name
  // TODO check for unknown shape
  auto source_fname = _pconfig.source.model_file;

  INFO(l) << "save_outputs() loading file: " << source_fname << std::endl;
  auto module = import_circle(source_fname);

  const auto output_nodes = loco::output_nodes(module->graph());
  for (uint32_t i = 0; i < module->graph()->outputs()->size(); i++)
  {
    const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);

    auto output_name = output_node->name();
    INFO(l) << "save_outputs() save output node: " << output_name << std::endl;
    assert(_data_stage.find(output_name) != _data_stage.end());

    auto tensor_data = _data_stage[output_name];
    auto output_filename = output_file + std::to_string(i);

    write_file(output_filename, tensor_data.data(), tensor_data.size());
    save_shape(output_filename + ".shape", output_node);
  }
}

} // namespace prunner
