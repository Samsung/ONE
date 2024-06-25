/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SingleModelExecutors.h"

#include "../backend/builtin/EdgeTensor.h"
#include "../backend/builtin/UserTensor.h"
#include "IPermuteFunction.h"

#include <array>

namespace onert
{
namespace exec
{

void SingleModelExecutors::emplace(const ir::ModelIndex &, const ir::SubgraphIndex &subg_index,
                                   std::unique_ptr<IExecutor> exec)
{
  _executors.emplace(subg_index, std::move(exec));
}

IExecutor *SingleModelExecutors::at(const ir::ModelIndex &,
                                    const ir::SubgraphIndex &subg_index) const
{
  return _executors.at(subg_index).get();
}

uint32_t SingleModelExecutors::inputSize() const { return entryExecutor()->inputSize(); }

uint32_t SingleModelExecutors::outputSize() const { return entryExecutor()->outputSize(); }

const ir::OperandInfo &SingleModelExecutors::inputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->inputInfo(index.value());
}

const ir::OperandInfo &SingleModelExecutors::outputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->outputInfo(index.value());
}

void SingleModelExecutors::execute(const ExecutionContext &ctx)
{
  class PermuteLayer : public exec::IPermuteFunction
  {
  public:
    PermuteLayer(const std::vector<backend::ITensor *> &inputs,
                 const std::vector<backend::ITensor *> &outputs)
    {
      assert(inputs.size() == outputs.size());
      _src_tensors = inputs;
      _dst_tensors = outputs;
    }
    virtual ~PermuteLayer() {}
    void optimize() override {}
  };

  // Create Input/Output UserTensors
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> tensorpool;
  std::vector<std::unique_ptr<backend::builtin::EdgeTensor>> qtensorpool;

  // Vector for executor I/O
  std::vector<backend::IPortableTensor *> inputs(ctx.desc.inputs.size());
  std::vector<backend::IPortableTensor *> outputs(ctx.desc.outputs.size());

  // Vector for input quantization I/O
  std::vector<backend::ITensor *> input_tensors;
  std::vector<backend::ITensor *> input_qtensors;

  // Vector for output dequantization I/O
  std::vector<backend::ITensor *> output_qtensors;
  std::vector<backend::ITensor *> output_tensors;

  for (uint32_t i = 0; i < inputs.size(); i++)
  {
    auto &desc = ctx.desc.inputs[i];

    // Input is optional if buffer is nullptr, and optional input's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Input " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<const uint8_t *>(desc->buffer), desc->size));

    auto user_type = desc->info.typeInfo().type();
    auto &model_info = entryExecutor()->inputInfo(i).typeInfo();
    auto model_type = model_info.type();
    if (user_type != model_type && user_type == ir::DataType::FLOAT32)
    {
      auto quantized_info = desc->info;
      quantized_info.typeInfo(model_info);
      qtensorpool.emplace_back(std::make_unique<backend::builtin::EdgeTensor>(
        quantized_info, entryExecutor()->inputLayout(i)));
      qtensorpool.back()->allocate_buffer();

      input_tensors.push_back(tensorpool.back().get());
      input_qtensors.push_back(qtensorpool.back().get());
      inputs[i] = qtensorpool.back().get();
    }
    else
      inputs[i] = tensorpool.back().get();
  }

  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    auto &desc = ctx.desc.outputs[i];

    // Output is optional if buffer is nullptr, and optional output's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<const uint8_t *>(desc->buffer), desc->size));

    auto user_type = desc->info.typeInfo().type();
    auto &model_info = entryExecutor()->outputInfo(i).typeInfo();
    auto model_type = model_info.type();
    if (user_type != model_type && user_type == ir::DataType::FLOAT32)
    {
      auto quantized_info = desc->info;
      quantized_info.typeInfo(model_info);
      qtensorpool.emplace_back(std::make_unique<backend::builtin::EdgeTensor>(
        quantized_info, entryExecutor()->outputLayout(i)));
      qtensorpool.back()->allocate_buffer();

      output_qtensors.push_back(qtensorpool.back().get());
      output_tensors.push_back(tensorpool.back().get());
      outputs[i] = qtensorpool.back().get();
    }
    else
      outputs[i] = tensorpool.back().get();
  }

  if (input_tensors.size() > 0)
  {
    auto input_quantize_layer = PermuteLayer(input_tensors, input_qtensors);
    input_quantize_layer.prepare();
    input_quantize_layer.run();
  }

  entryExecutor()->execute(inputs, outputs, ctx.options);

  if (output_tensors.size() != 0)
  {
    auto output_dequantize_layer = PermuteLayer(output_qtensors, output_tensors);
    output_dequantize_layer.prepare();
    output_dequantize_layer.run();
  }

  // Get dynamic shape inference result
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    if (ctx.desc.outputs[i]->buffer == nullptr)
    {
      // Output is optional if buffer is nullptr
      continue;
    }

    ctx.desc.outputs[i]->info.shape(outputs[i]->getShape());
  }
}

} // namespace exec
} // namespace onert
