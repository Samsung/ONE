/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleEvalDiff.h"
#include "MetricPrinter.h"
#include "Tensor.h"

#include <foder/FileLoader.h>
#include <luci/Importer.h>

#include <stdexcept>

namespace
{

std::unique_ptr<luci::Module> import(const std::string &model_path)
{
  // Load model from the file
  foder::FileLoader loader{model_path};
  std::vector<char> model_data = loader.load();

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (not circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Failed to verify circle '" + model_path + "'");
  }

  auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (not module)
    throw std::runtime_error("Failed to load '" + model_path + "'");

  return module;
}

const std::vector<loco::Node *> inputs_of(const luci::Module *module)
{
  return loco::input_nodes(module->graph());
}

const std::vector<loco::Node *> outputs_of(const luci::Module *module)
{
  return loco::output_nodes(module->graph());
}

} // namespace

namespace circle_eval_diff
{

std::vector<std::shared_ptr<Tensor>> interpret(const luci::Module *module,
                                               const InputDataLoader::Data &data)
{
  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module);

  auto input_nodes = ::inputs_of(module);
  auto output_nodes = ::outputs_of(module);

  for (uint32_t input_idx = 0; input_idx < data.size(); input_idx++)
  {
    auto input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
    assert(input_node->index() == input_idx);

    auto input_data = data.at(input_idx);
    interpreter->writeInputTensor(input_node, input_data.buffer(), input_data.byte_size());
  }

  interpreter->interpret();

  std::vector<std::shared_ptr<Tensor>> outputs;
  for (uint32_t output_idx = 0; output_idx < output_nodes.size(); output_idx++)
  {
    auto output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[output_idx]);
    assert(output_node->index() == output_idx);

    auto tensor = createEmptyTensor(output_node);
    interpreter->readOutputTensor(output_node, tensor->buffer(), tensor->byte_size());
    outputs.emplace_back(tensor);
  }

  return outputs;
}

CircleEvalDiff::CircleEvalDiff(std::unique_ptr<Context> &&ctx) : _ctx(std::move(ctx))
{
  // DO NOTHING
}

CircleEvalDiff::~CircleEvalDiff() = default;

void CircleEvalDiff::init()
{
  _first_module = import(_ctx->first_model_path);
  _second_module = import(_ctx->second_model_path);

  // Set metric
  std::unique_ptr<MetricPrinter> metric;
  for (auto metric : _ctx->metric)
  {
    switch (metric)
    {
      case Metric::MAE:
      {
        _metrics.emplace_back(std::make_unique<MAEPrinter>());
        break;
      }
      case Metric::MAPE:
      {
        _metrics.emplace_back(std::make_unique<MAPEPrinter>());
        break;
      }
      case Metric::MPEIR:
      {
        _metrics.emplace_back(std::make_unique<MPEIRPrinter>());
        break;
      }
      default:
        throw std::runtime_error("Unsupported metric.");
    }
    _metrics.back()->init(_first_module.get(), _second_module.get());
  }
}

void CircleEvalDiff::evalDiff(const InputDataLoader *first, const InputDataLoader *second) const
{
  for (uint32_t data_idx = 0; data_idx < first->size(); data_idx++)
  {
    std::cout << "Evaluating " << data_idx << "'th data" << std::endl;

    auto first_data = first->get(data_idx);
    auto second_data = second->get(data_idx);

    auto first_output = interpret(_first_module.get(), first_data);
    auto second_output = interpret(_second_module.get(), second_data);

    for (auto &metric : _metrics)
    {
      metric->accumulate(first_output, second_output);
    }
  }

  for (auto &metric : _metrics)
  {
    std::cout << metric.get() << std::endl;
  }
}

const std::vector<loco::Node *> CircleEvalDiff::first_module_inputs(void) const
{
  return loco::input_nodes(_first_module->graph());
}

const std::vector<loco::Node *> CircleEvalDiff::second_module_inputs(void) const
{
  return loco::input_nodes(_second_module->graph());
}

} // namespace circle_eval_diff
