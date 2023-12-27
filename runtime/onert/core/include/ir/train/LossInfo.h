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

#ifndef __ONERT_IR_TRAIN_LOSS_INFO_H__
#define __ONERT_IR_TRAIN_LOSS_INFO_H__

#include "LossCode.h"

namespace onert
{
namespace ir
{
namespace train
{

enum class LossReductionType
{
  Invalid,          //< Invalid
  Auto,             //< Auto
  SumOverBatchSize, //< SumOverBatchSize loss reduction type
  Sum,              //< Sum loss reduction type
};

struct CategoricalCrossentropyParam
{
  int32_t axis;
  float label_smoothing;
};

struct LossInfo
{
  LossCode loss_code;
  LossReductionType reduction_type;
  union LossParam {
    CategoricalCrossentropyParam cce;
  } loss_param;

  LossInfo()
    : loss_code{LossCode::Invalid}, reduction_type{LossReductionType::Invalid}, loss_param{-1, 0.0f}
  {
  }
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_LOSS_INFO_H__
