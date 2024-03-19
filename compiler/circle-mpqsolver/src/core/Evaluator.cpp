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

#include "Evaluator.h"

#include "core/DataProvider.h"

#include <luci/IR/DataTypeHelper.h>

#include <luci_interpreter/Interpreter.h>

#include <dio_hdf5/HDF5Importer.h>

using namespace mpqsolver::core;

using Shape = std::vector<loco::Dimension>;

namespace
{

using namespace luci;

template <typename NodeT> size_t get_tensor_size(const NodeT *node)
{
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

WholeOutput compute_outputs(const luci::Module *module, const DataProvider *data_provider)
{
  if (data_provider == nullptr)
  {
    throw std::runtime_error("No data");
  }

  const auto num_records = data_provider->numSamples();
  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");
  const auto input_nodes = loco::input_nodes(module->graph());
  const auto num_inputs = input_nodes.size();

  WholeOutput dataset_output;

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(module);
  for (uint32_t record_idx = 0; record_idx < num_records; record_idx++)
  {
    if (num_inputs != data_provider->numInputs(record_idx))
      throw std::runtime_error("Wrong number of inputs.");
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      assert(input_node->index() == input_idx);

      InputData input_data(get_tensor_size(input_node));
      data_provider->getSampleInput(record_idx, input_idx, input_data);

      interpreter.writeInputTensor(input_node, input_data.data().data(), input_data.data().size());
    }

    interpreter.interpret();

    Output nn_output;

    // Get output.
    const auto output_nodes = loco::output_nodes(module->graph());
    for (size_t i = 0; i < module->graph()->outputs()->size(); i++)
    {
      const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);
      Buffer output_data(get_tensor_size(output_node));
      interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());
      // output
      nn_output.push_back(output_data);
    }
    dataset_output.push_back(nn_output);
  }

  return dataset_output;
}

} // namespace

DatasetEvaluator::DatasetEvaluator(const luci::Module *ref_module, const DataProvider &provider,
                                   const ErrorMetric &metric)
  : _ref_module(ref_module), _provider(&provider), _metric(&metric)
{
  _ref_output = compute_outputs(_ref_module, _provider);
}

void DatasetEvaluator::validate(const luci::Module *trgt_fq_module) const
{
  const auto output_nodes = loco::output_nodes(trgt_fq_module->graph());
  for (size_t out_index = 0; out_index < output_nodes.size(); ++out_index)
  {
    const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[out_index]);
    loco::DataType out_dtype = output_node->dtype();
    if (out_dtype != loco::DataType::FLOAT32)
      throw std::runtime_error("Unsupported output dtype " + output_node->name());
  }
}

float DatasetEvaluator::evaluate(const luci::Module *trgt_fq_module) const
{
  if (trgt_fq_module == nullptr)
    throw std::runtime_error("Invalid target module");

  if (_metric == nullptr)
    throw std::runtime_error("Invalid metric");

  validate(trgt_fq_module);

  const WholeOutput &cur_output = compute_outputs(trgt_fq_module, _provider);
  float error = _metric->compute(_ref_output, cur_output);
  return error;
}
