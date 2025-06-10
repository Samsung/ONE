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
#include "InputDataLoader.h"
#include "MetricPrinter.h"
#include "Tensor.h"

#include <foder/FileLoader.h>
#include <luci/ImporterEx.h>

#include <stdexcept>

namespace
{

bool same_shape(const luci::CircleNode *a, const luci::CircleNode *b)
{
  if (a->rank() != b->rank())
    return false;

  for (uint32_t i = 0; i < a->rank(); i++)
  {
    if (not(a->dim(i) == b->dim(i)))
      return false;
  }

  return true;
}

bool same_dtype(const luci::CircleNode *a, const luci::CircleNode *b)
{
  return a->dtype() == b->dtype();
}

std::unique_ptr<luci::Module> import(const std::string &model_path)
{
  luci::ImporterEx importer;
  auto module = importer.importVerifyModule(model_path);

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

void writeDataToFile(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

void checkOutputs(const luci::Module *first, const luci::Module *second)
{
  const auto first_output = outputs_of(first);
  const auto second_output = outputs_of(second);

  if (first_output.size() != second_output.size())
    throw std::runtime_error("Models have different output counts");

  for (uint32_t i = 0; i < first_output.size(); i++)
  {
    const auto first_node = loco::must_cast<luci::CircleNode *>(first_output[i]);
    const auto second_node = loco::must_cast<luci::CircleNode *>(second_output[i]);

    if (not same_shape(first_node, second_node))
      throw std::runtime_error("Output shape mismatch (" + first_node->name() + ", " +
                               second_node->name() + ")");

    if (not same_dtype(first_node, second_node))
      throw std::runtime_error("Output dtype mismatch (" + first_node->name() + ", " +
                               second_node->name() + ")");
  }
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

  // Check modules have the same output signature (dtype/shape)
  // Exception will be thrown if they have different signature
  checkOutputs(_first_module.get(), _second_module.get());

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
      case Metric::MTOP1:
      {
        _metrics.emplace_back(std::make_unique<TopKMatchPrinter>(1));
        break;
      }
      case Metric::MTOP5:
      {
        _metrics.emplace_back(std::make_unique<TopKMatchPrinter>(5));
        break;
      }
      case Metric::MSE:
      {
        _metrics.emplace_back(std::make_unique<MSEPrinter>());
        break;
      }
      default:
        throw std::runtime_error("Unsupported metric.");
    }
    _metrics.back()->init(_first_module.get(), _second_module.get());
  }
}

void CircleEvalDiff::evalDiff(void) const
{
  auto first_input_loader = circle_eval_diff::makeDataLoader(
    _ctx->first_input_data_path, _ctx->input_format, ::inputs_of(_first_module.get()));
  auto second_input_loader = circle_eval_diff::makeDataLoader(
    _ctx->second_input_data_path, _ctx->input_format, ::inputs_of(_second_module.get()));

  for (uint32_t data_idx = 0; data_idx < first_input_loader->size(); data_idx++)
  {
    std::cout << "Evaluating " << data_idx << "'th data" << std::endl;

    auto first_data = first_input_loader->get(data_idx);
    auto second_data = second_input_loader->get(data_idx);

    auto first_output = interpret(_first_module.get(), first_data);
    auto second_output = interpret(_second_module.get(), second_data);

    for (auto &metric : _metrics)
    {
      metric->accumulate(first_output, second_output);
    }

    if (_ctx.get()->output_prefix.empty())
      continue;

    for (uint32_t i = 0; i < first_output.size(); i++)
    {
      auto out = first_output[i];
      writeDataToFile(_ctx.get()->output_prefix + "." + std::to_string(data_idx) + ".first.output" +
                        std::to_string(i),
                      (char *)(out->buffer()), out->byte_size());
    }
    for (uint32_t i = 0; i < second_output.size(); i++)
    {
      auto out = second_output[i];
      writeDataToFile(_ctx.get()->output_prefix + "." + std::to_string(data_idx) +
                        ".second.output" + std::to_string(i),
                      (char *)(out->buffer()), out->byte_size());
    }
  }

  for (auto &metric : _metrics)
  {
    std::cout << metric.get() << std::endl;
  }
}

} // namespace circle_eval_diff
