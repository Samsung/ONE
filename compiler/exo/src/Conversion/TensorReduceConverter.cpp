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

#include "TensorReduceConverter.h"

#include "Dialect/IR/TFLNodes.h"
#include "Check.h"

#include <oops/InternalExn.h>

#include <loco.h>
#include <loco/Service/ShapeInference.h>

namespace
{

/**
 * @brief  Convert given TensorReduce as TFLMean
 *
 * <Before>
 *    In --- loco::TensorReduce --- Out(s)
 *
 * <After>
 *    In -------- locoex::TFLMean --- Out(s)
 *                /
 *    TFLConst ---
 *    (reduction indices)
 */
bool convert_as_mean(loco::TensorReduce *origin)
{
  EXO_ASSERT(origin->func() == loco::ReduceFunc::Mean, "func should be Mean for this helper");
  EXO_ASSERT(origin->input(), "TensorReduce has no input");

  auto *graph = origin->graph();

  // Make reduction indicies TFLConst node
  auto reduction = graph->nodes()->create<locoex::TFLConst>();
  {
    auto input_rank = loco::shape_get(origin->input()).as<loco::TensorShape>().rank();

    std::vector<int32_t> red_vec;
    for (uint32_t axis = 0; axis < input_rank; ++axis)
      if (origin->axes()->defined(axis))
        red_vec.push_back(static_cast<int32_t>(axis));

    const loco::DataType S32 = loco::DataType::S32;

    reduction->dtype(S32);
    reduction->rank(1);
    reduction->dim(0) = red_vec.size();
    reduction->size<S32>(red_vec.size());
    for (uint32_t i = 0; i < red_vec.size(); ++i)
      reduction->at<S32>(i) = red_vec.at(i);
  }

  // Make TFLMean node to replace
  auto mean = graph->nodes()->create<locoex::TFLMean>();
  mean->input(origin->input());
  mean->reduction_indices(reduction);
  mean->keep_dims(true); // Canonical TensorReduce always keep dimensions

  // replace canonical node
  loco::replace(origin).with(mean);
  origin->input(nullptr);

  return true;
}

} // namespace

namespace exo
{

bool TensorReduceConverter::convert(loco::TensorReduce *origin)
{
  if (origin->func() == loco::ReduceFunc::Mean)
    return convert_as_mean(origin);
  else
    INTERNAL_EXN_V("Unsupported ReduceFunc", oops::to_uint32(origin->func()));
}

} // namespace exo
