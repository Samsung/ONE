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

#ifndef __ONERT_BACKEND_TRAIN_OPS_PADLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_PADLAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include <exec/train/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

// Note, this is pad with mode=`CONSTANT`: it doesn't support `REFLECT` and
// `SYMMETRIC`
class PadLayer : public ::onert::exec::train::ITrainableFunction
{
public:
  PadLayer();

public:
  template <typename T> void padImpl(const T *constant_value_data);
  template <typename T> void depad();

  void configure(const IPortableTensor *input, IPortableTensor *output,
                         const int32_t *padData, int32_t padRank, const void *constantValueData,
                         IPortableTensor *back_prop_input, const IPortableTensor *back_prop_output);
  void forward(bool training) override;
  void backward() override;

private:
  const IPortableTensor *_input;
  IPortableTensor *_output;

  int32_t _padData[8];
  int32_t _padRank;
  ConstDataPtr _constantValueData;

  IPortableTensor *_back_prop_input;
  const IPortableTensor *_back_prop_output;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_PADLAYER_H__
