/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConstGenConverter.h"

#include "Dialect/IR/TFLNodes.h"
#include "Check.h"

#include <loco.h>

#include <oops/InternalExn.h>

namespace exo
{

bool ConstGenConverter::convert(loco::ConstGen *constgen)
{
  auto *graph = constgen->graph();

  auto tfl_const = graph->nodes()->create<locoex::TFLConst>();
  {
    if (constgen->dtype() == loco::DataType::FLOAT32)
    {
      tfl_const->dtype(loco::DataType::FLOAT32);

      tfl_const->rank(constgen->rank());
      for (uint32_t axis = 0; axis < constgen->rank(); axis++)
        tfl_const->dim(axis) = constgen->dim(axis);

      auto size = constgen->size<loco::DataType::FLOAT32>();
      tfl_const->size<loco::DataType::FLOAT32>(size);

      for (uint32_t i = 0; i < size; ++i)
      {
        tfl_const->at<loco::DataType::FLOAT32>(i) = constgen->at<loco::DataType::FLOAT32>(i);
      }
    }
    else
      INTERNAL_EXN_V("Unsupported DataType", oops::to_uint32(constgen->dtype()));
  }

  loco::replace(constgen).with(tfl_const);

  return true;
}

} // namespace exo
