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

#ifndef __ONERT_BACKEND_TRAIN_OPS_POOLLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_POOLLAYER_H__

#include <ops/PoolLayer.h>

#include <exec/train/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

/**
 * This is to register the pair of (forward, backward) training kernel.
 */
class TrainingKernelRegistry
{
public:
  virtual void forward(const IPortableTensor *in, IPortableTensor *out) = 0;
  virtual void backward(const IPortableTensor *back_prop_out, IPortableTensor *back_prop_in) = 0;
  TrainingKernelRegistry() = default;
  virtual ~TrainingKernelRegistry() = default;
};

enum class PoolType
{
  kMax,
};

class PoolLayer : public ::onert::exec::train::ITrainableFunction, public cpu::ops::PoolLayer
{
public:
  PoolLayer();

public:
  void configure(const IPortableTensor *input, const uint32_t paddingLeft,
                 const uint32_t paddingRight, const uint32_t paddingTop,
                 const uint32_t paddingBottom, const uint32_t strideWidth,
                 const uint32_t strideHeight, const uint32_t kernelWidth,
                 const uint32_t kernelHeight, const ir::Activation activation,
                 IPortableTensor *output, const PoolType op_type, IPortableTensor *back_prop_input,
                 const IPortableTensor *back_prop_output);

  void forward(bool training) override;
  void backward() override;

private:
  IPortableTensor *_back_prop_input;
  const IPortableTensor *_back_prop_output;

  std::unique_ptr<TrainingKernelRegistry> _kernel;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_POOLLAYER_H__
