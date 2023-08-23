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

#ifdef ENABLE_TRAINING

#include "TrainingGraph.h"

#include "kernels/KernelBuilder.h"

#include <unordered_map>

namespace luci_interpreter
{
namespace training
{

Status TrainingGraph::saveLabelDataAsBackDerivative(CircleReader *reader,
                                                    TrainableWeightStorage *storage,
                                                    const uint8_t *label_train_data)
{
  Status status;

  const auto graph_outputs = reader->outputs();
  assert(graph_outputs.size() == 1);
  if (graph_outputs.size() != 1)
    return Error;

  const circle::Tensor *output_graph_tensor = reader->tensors()[graph_outputs[0]];

  uint8_t *output_data = nullptr;
  status = _gradient_calculation_storage.getDataByTensor(output_graph_tensor, &output_data);
  if (status != Ok)
    return status;

  assert(output_data != nullptr);
  if (output_data == nullptr)
    return Error;

  const auto tensor_size = Tensor::num_elements(output_graph_tensor);
  const auto tensor_type = Tensor::element_type(output_graph_tensor);

  switch (tensor_type)
  {
    case DataType::FLOAT32:
    {
      float *casted_output_data = reinterpret_cast<float *>(output_data);
      const float *casted_label_data = reinterpret_cast<const float *>(label_train_data);

      // For MSE
      for (int i = 0; i < tensor_size; ++i)
        casted_output_data[i] = casted_output_data[i] - casted_label_data[i];

      break;
    }
    default:
    {
      assert(false && "Unsupported type");
      return Error;
    }
  }

  return Ok;
}

Status TrainingGraph::computeGradients(const TrainingSettings &settings,
                                       TrainableWeightStorage *storage, CircleReader *reader,
                                       const uint8_t *label_train_data)
{
  assert(settings.number_of_last_trainable_layers != 0);

  const int last_op_pos = reader->operators().size() - 1;
  const int last_train_op_pos = settings.number_of_last_trainable_layers > 0
                                  ? last_op_pos - settings.number_of_last_trainable_layers
                                  : -1;
  Status status;

  // Save label_data as gradient to output tensor
  status = saveLabelDataAsBackDerivative(reader, storage, label_train_data);
  if (status != Ok)
    return status;

  for (auto op_pos = last_op_pos; op_pos > last_train_op_pos; --op_pos)
  {
    const auto op = reader->operators().at(op_pos);
    const auto opcode = reader->builtin_code(op);

    status = kernel_train.train_kernel(op, opcode, reader, &_gradient_calculation_storage, settings,
                                       storage, true /* compute gradient mode */);

    if (status != Ok)
      return status;
  }

  _gradient_calculation_storage.clearComputedData();

  return Ok;
}

Status TrainingGraph::updateWeights(const TrainingSettings &settings,
                                    TrainableWeightStorage *storage, CircleReader *reader)
{
  const int last_op_pos = reader->operators().size() - 1;
  const int last_train_op_pos = settings.number_of_last_trainable_layers > 0
                                  ? last_op_pos - settings.number_of_last_trainable_layers
                                  : -1;
  Status status;
  for (auto op_pos = last_op_pos; op_pos > last_train_op_pos; --op_pos)
  {
    const auto op = reader->operators().at(op_pos);
    const auto opcode = reader->builtin_code(op);

    status = kernel_train.train_kernel(op, opcode, reader, &_gradient_calculation_storage, settings,
                                       storage, false /* update weights mode */);

    assert(status == Ok);

    if (status != Ok)
    {
      return status;
    }
  }
  _gradient_calculation_storage.clearComputedData();
  _gradient_calculation_storage.clearComputedGradients();

  return Ok;
}

} // namespace training
} // namespace luci_interpreter

#endif // ENABLE_TRAINING
