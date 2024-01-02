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

#ifndef __ONERT_BACKEND_TRAIN_OPS_MEANLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_MEANLAYER_H__

#include <ops/MeanLayer.h>
#include <backend/IPortableTensor.h>

#include <exec/train/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class MeanLayer : public ::onert::exec::train::ITrainableFunction, public cpu::ops::MeanLayer
{
public:
  MeanLayer();

public:
  void configure(const IPortableTensor *input, const IPortableTensor *axes, IPortableTensor *output,
                 bool keep_dims, IPortableTensor *back_prop_input,
                 const IPortableTensor *back_prop_output);
  void forward(bool training) override;
  void backward() override;

private:
  IPortableTensor *_back_prop_input;
  const IPortableTensor *_back_prop_output;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_MEANLAYER_H__
