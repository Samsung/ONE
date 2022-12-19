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
#include "ModuleCloner.h"

#include <luci_interpreter/Interpreter.h>

#include <dio_hdf5/HDF5Importer.h>

#include <cmath>
#include <iostream>

using namespace mpqsolver;

using Shape = std::vector<loco::Dimension>;

namespace
{

using namespace luci;

template <typename NodeT> size_t get_tensor_size(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

} // namespace

DatasetOutput compute_outputs(const luci::Module *module, const std::string &h5file)
{
  dio::hdf5::HDF5Importer importer{h5file};
  importer.importGroup("value");

  bool is_raw_data = importer.isRawData();

  const auto num_records = importer.numData();
  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");
  const auto input_nodes = loco::input_nodes(module->graph());
  const auto num_inputs = input_nodes.size();

  DatasetOutput dataset_output;

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(module);
  for (int32_t record_idx = 0; record_idx < num_records; record_idx++)
  {
    if (num_inputs != importer.numInputs(record_idx))
      throw std::runtime_error("Wrong number of inputs.");
    for (int32_t input_idx = 0; input_idx < num_inputs; input_idx++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      assert(input_node->index() == input_idx);

      std::vector<char> input_data(get_tensor_size(input_node));

      if (!is_raw_data)
      {
        loco::DataType dtype;
        Shape shape;
        importer.readTensor(record_idx, input_idx, &dtype, &shape, input_data.data());
      }
      else
      {
        // Skip type/shape check for raw data
        importer.readTensor(record_idx, input_idx, input_data.data());
      }

      interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());
    }

    interpreter.interpret();

    NNOutput nn_output;

    // Get output.
    const auto output_nodes = loco::output_nodes(module->graph());
    for (int i = 0; i < module->graph()->outputs()->size(); i++)
    {
      const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);
      ElementaryOutput output_data(get_tensor_size(output_node));
      interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());
      // output
      nn_output.push_back(output_data);
    }
    dataset_output.push_back(nn_output);
  }

  return dataset_output;
}

DatasetEvaluator::DatasetEvaluator(const luci::Module *ref_module, const std::string &h5file)
  : _ref_module(ref_module), _h5file(h5file)
{
  _ref_output = compute_outputs(_ref_module, _h5file);
}

float DatasetEvaluator::evaluate(const luci::Module *trgt_fq_module) const
{
  float accuracy = 0.f;
  size_t output_size = 0;
  const DatasetOutput &cur_output = compute_outputs(trgt_fq_module, _h5file);
  for (size_t sample_index = 0; sample_index < _ref_output.size(); ++sample_index)
  {
    // Get output.
    const auto output_nodes = loco::output_nodes(trgt_fq_module->graph());
    for (size_t out_index = 0; out_index < _ref_output[sample_index].size(); ++out_index)
    {
      const auto *output_node =
        loco::must_cast<const luci::CircleOutput *>(output_nodes[out_index]);
      loco::DataType out_type = output_node->dtype();
      const ElementaryOutput &ref_elementary = _ref_output[sample_index][out_index];
      const ElementaryOutput &cur_elementary = cur_output[sample_index][out_index];
      size_t cur_size = ref_elementary.size() / loco::size(out_type);

      switch (out_type)
      {
        case loco::DataType::FLOAT32: {
          const float *ref_floats = reinterpret_cast<const float *>(ref_elementary.data());
          const float *cur_floats = reinterpret_cast<const float *>(cur_elementary.data());
          for (size_t index = 0; index < cur_size; index++)
          {
            float ref_value = *(ref_floats + index);
            float cur_value = *(cur_floats + index);
            accuracy += std::fabs(ref_value - cur_value);
          }
          output_size += cur_size;
        }
        break;
        default:
          throw std::runtime_error("Unknown out_type");
      }
    }
  }

  accuracy /= output_size;
  return accuracy;
}
