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

#include "IPermuteFunction.h"
#include "../backend/builtin/UserTensor.h"

// TODO Share with MultiModelExecutors and remove
namespace
{

class EdgeTensor : public onert::backend::IPortableTensor
{
public:
  EdgeTensor(const onert::ir::OperandInfo &info, onert::ir::Layout layout)
    : onert::backend::IPortableTensor(info), _layout{layout}, _buffer{nullptr}, _ref_count{0}
  {
  }
  ~EdgeTensor() = default;

  uint8_t *buffer() const override { return _buffer.get(); }
  onert::ir::Layout layout() const override { return _layout; }
  void set_dynamic() override { _info.setDynamic(); }
  bool applyShape(const onert::ir::Shape &new_shape) override
  {
    bool previously_dynamic = is_dynamic();
    if (!previously_dynamic || _buffer == nullptr)
    {
      // Always set shape - when buffer with same size was already allocated, shape could differ
      setShape(new_shape);
      set_dynamic();
      const auto total_size = get_info().total_size();
      _buffer = std::make_unique<uint8_t[]>(total_size);
    }
    else
    {
      auto previous_size = total_size();
      auto new_size = new_shape.num_elements() * onert::ir::sizeOfDataType(data_type());
      if (previous_size != new_size)
      {
        setShape(new_shape);
        set_dynamic();
        const auto total_size = get_info().total_size();
        _buffer = std::make_unique<uint8_t[]>(total_size);
      }
      else
      { // when buffer with same size was already allocated, shape could differ
        setShape(new_shape);
      }
    }
    return true;
  }

  void setShape(const onert::ir::Shape &new_shape) override { _info.shape(new_shape); }

  void allocate_buffer()
  {
    const auto total_size = _info.total_size();
    _buffer = std::make_unique<uint8_t[]>(total_size);
    _ref_count = 1;
  }

  void increase_ref() { _ref_count++; }

  void decrease_ref()
  {
    assert(_ref_count > 0);
    _ref_count--;
    if (_ref_count == 0)
    {
      _buffer.reset();
    }
  }

private:
  onert::ir::Layout _layout;
  std::unique_ptr<uint8_t[]> _buffer;
  int32_t _ref_count;
};

class PermuteLayer : public onert::exec::IPermuteFunction
{
public:
  PermuteLayer(const std::vector<onert::backend::ITensor *> &inputs,
               const std::vector<onert::backend::ITensor *> &outputs)
  {
    assert(inputs.size() == outputs.size());
    _src_tensors = inputs;
    _dst_tensors = outputs;
  }
  virtual ~PermuteLayer() {}
  void optimize() override {}
};

} // namespace

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

  // Vector for output dequantization I/O
  std::vector<backend::ITensor *> output_qtensors;
  std::vector<backend::ITensor *> output_tensors;

  // Prepare UserTensor and EdgeTensor for input quantization
  for (uint32_t i = 0; i < inputs.size(); i++)
  {
    auto &desc = ctx.desc.inputs[i];

    // Input is optional if buffer is nullptr, and optional input's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Input " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, const_cast<uint8_t *>(static_cast<const uint8_t *>(desc->buffer)),
      desc->size));

    auto user_type = desc->info.typeInfo().type();
    auto &model_info = entryExecutor()->inputInfo(i).typeInfo();
    auto model_type = model_info.type();
    if (user_type != model_type && user_type == ir::DataType::FLOAT32)
    {
      auto quantized_info = desc->info;
      quantized_info.typeInfo(model_info);
      qtensorpool.emplace_back(
        std::make_unique<EdgeTensor>(quantized_info, entryExecutor()->inputLayout(i)));
      qtensorpool.back()->allocate_buffer();

      input_tensors.push_back(tensorpool.back().get());
      input_qtensors.push_back(qtensorpool.back().get());
      inputs[i] = qtensorpool.back().get();
    }
    else
      inputs[i] = tensorpool.back().get();
  }

  // Prepare UserTensor and EdgeTensor for output dequantization
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    auto &desc = ctx.desc.outputs[i];

    // Output is optional if buffer is nullptr, and optional output's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<uint8_t *>(desc->buffer), desc->size));

    auto user_type = desc->info.typeInfo().type();
    auto &model_info = entryExecutor()->outputInfo(i).typeInfo();
    auto model_type = model_info.type();
    if (user_type != model_type && user_type == ir::DataType::FLOAT32)
    {
      auto quantized_info = desc->info;
      quantized_info.typeInfo(model_info);
      qtensorpool.emplace_back(
        std::make_unique<EdgeTensor>(quantized_info, entryExecutor()->outputLayout(i)));
      qtensorpool.back()->allocate_buffer();

      output_qtensors.push_back(qtensorpool.back().get());
      output_tensors.push_back(tensorpool.back().get());
      outputs[i] = qtensorpool.back().get();
    }
    else
      outputs[i] = tensorpool.back().get();
  }

  // Run quantization
  if (input_tensors.size() > 0)
  {
    auto input_quantize_layer = PermuteLayer(input_tensors, input_qtensors);
    input_quantize_layer.prepare();
    input_quantize_layer.run();
  }

  // Executor
  entryExecutor()->execute(inputs, outputs, ctx.options);

  // Run dequantization
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
