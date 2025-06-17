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

#include "EdgeTensor.h"
#include "IPermuteFunction.h"
#include "../backend/builtin/UserTensor.h"
#include "../backend/builtin/IOTensor.h"

namespace onert::exec
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

const void *SingleModelExecutors::outputBuffer(const ir::IOIndex &index) const
{
  return static_cast<const void *>(entryExecutor()->outputBuffer(index.value()));
}

const backend::IPortableTensor *SingleModelExecutors::outputTensor(const ir::IOIndex &index) const
{
  return entryExecutor()->outputTensor(index.value());
}

void SingleModelExecutors::execute(const ExecutionContext &ctx)
{
  // UserTensor for Input/Output
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> tensorpool;

  // EdgeTensor for Input Quantization / Output Dequantization
  std::vector<std::unique_ptr<EdgeTensor>> qtensorpool;

  // Input/Output Tensor vector for executor
  std::vector<backend::IPortableTensor *> inputs(ctx.desc.inputs.size());
  std::vector<backend::IPortableTensor *> outputs(ctx.desc.outputs.size());

  // Vector for input quantization I/O
  std::vector<backend::ITensor *> input_tensors;
  std::vector<backend::ITensor *> input_qtensors;
  std::vector<ir::PermuteType> input_permute_types;

  // Vector for output dequantization I/O
  std::vector<backend::ITensor *> output_qtensors;
  std::vector<backend::ITensor *> output_tensors;
  std::vector<ir::PermuteType> output_permute_types;

  // Prepare UserTensor and EdgeTensor for input quantization
  for (uint32_t i = 0; i < inputs.size(); i++)
  {
    auto &desc = ctx.desc.inputs[i];

    // Input is optional if buffer is nullptr, and optional input's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Input " + std::to_string(i) + "'s buffer is not set."};

    // TODO: Create UserTensor only that will be set into IOTensor
    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, const_cast<uint8_t *>(static_cast<const uint8_t *>(desc->buffer)),
      desc->size));

    auto user_type = desc->info.typeInfo().type();
    auto &model_info = entryExecutor()->inputInfo(i).typeInfo();
    auto model_type = model_info.type();
    if ((user_type != model_type && user_type == ir::DataType::FLOAT32) ||
        (desc->layout == ir::Layout::NCHW))
    {
      auto quantized_info = desc->info;
      quantized_info.typeInfo(model_info);
      qtensorpool.emplace_back(
        std::make_unique<EdgeTensor>(quantized_info, entryExecutor()->inputLayout(i)));
      qtensorpool.back()->allocate_buffer();

      input_tensors.push_back(tensorpool.back().get());
      input_qtensors.push_back(qtensorpool.back().get());
      inputs[i] = qtensorpool.back().get();
      if (desc->layout == ir::Layout::NCHW)
        input_permute_types.push_back(ir::PermuteType::NCHW_TO_NHWC);
      else
        input_permute_types.push_back(ir::PermuteType::COPY);
    }
    else
      inputs[i] = tensorpool.back().get();
  }

  // Prepare UserTensor and EdgeTensor for output dequantization
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    auto &desc = ctx.desc.outputs[i];
    bool skip_set_output =
      dynamic_cast<const backend::builtin::IOTensor *>(outputTensor(ir::IOIndex{i}))
        ->hasBackendTensor();

    // Output is optional if buffer is nullptr, and optional output's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0) &&
        !skip_set_output)
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<uint8_t *>(desc->buffer), desc->size));

    auto user_type = desc->info.typeInfo().type();
    auto &model_info = entryExecutor()->outputInfo(i).typeInfo();
    auto model_type = model_info.type();
    if ((user_type != model_type && user_type == ir::DataType::FLOAT32) ||
        (desc->layout == ir::Layout::NCHW))
    {
      if (skip_set_output)
        std::runtime_error("When outputs are allocated internally, backend-aware quantization is "
                           "not yet supported.");

      auto quantized_info = desc->info;
      quantized_info.typeInfo(model_info);
      qtensorpool.emplace_back(
        std::make_unique<EdgeTensor>(quantized_info, entryExecutor()->outputLayout(i)));
      qtensorpool.back()->allocate_buffer();

      output_qtensors.push_back(qtensorpool.back().get());
      output_tensors.push_back(tensorpool.back().get());
      outputs[i] = qtensorpool.back().get();
      if (desc->layout == ir::Layout::NCHW)
        output_permute_types.push_back(ir::PermuteType::NHWC_TO_NCHW);
      else
        output_permute_types.push_back(ir::PermuteType::COPY);
    }
    else
      outputs[i] = tensorpool.back().get();
  }

  // Run quantization
  if (input_tensors.size() > 0)
  {
    auto input_quantize_layer = PermuteLayer(input_tensors, input_qtensors, input_permute_types);
    input_quantize_layer.prepare();
    input_quantize_layer.run();
  }

  // Executor
  entryExecutor()->execute(inputs, outputs, ctx.options);

  // Run dequantization
  if (output_tensors.size() != 0)
  {
    auto output_dequantize_layer =
      PermuteLayer(output_qtensors, output_tensors, output_permute_types);
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

} // namespace onert::exec
