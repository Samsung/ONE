/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_OPS_BINARYARITHMETICLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_BINARYARITHMETICLAYER_H__

#include <ops/BinaryArithmeticLayer.h>
#include <backend/IPortableTensor.h>

#include "../Tensor.h"
#include <exec/train/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

enum class ArithmeticType
{
  kAdd,
  kSub,
  kMul,
  kDiv,
};

class BinaryArithmeticLayer : public ::onert::exec::train::ITrainableFunction,
                              public cpu::ops::BinaryArithmeticLayer
{
public:
  BinaryArithmeticLayer();

public:
  void configure(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
                 IPortableTensor *back_prop_lhs, IPortableTensor *back_prop_rhs,
                 const IPortableTensor *back_prop_output, const ir::Activation activation,
                 const ArithmeticType arithmetic_type);
  void forward(bool training) override;
  void backward() override;

private:
  IPortableTensor *_back_prop_lhs;
  IPortableTensor *_back_prop_rhs;
  const IPortableTensor *_back_prop_output;

  ArithmeticType _arithmetic_type;
  ir::Activation _activation;
  std::unique_ptr<BackPropTensor> _act_back_prop_output;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_BINARYARITHMETICLAYER_H__
