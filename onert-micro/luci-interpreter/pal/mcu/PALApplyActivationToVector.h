/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_PAL_APPLY_ACTIVATION_TO_VECTOR_H
#define LUCI_INTERPRETER_PAL_APPLY_ACTIVATION_TO_VECTOR_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include "tensorflow/lite/c/builtin_op_data.h"

namespace luci_interpreter_pal
{

// Dynamic (non-fused) activation functor. perhaps it is worth having
// template instantiation?
// TODO(aselle): Make this more efficient by pulling the switch to conv_eval
// using template inlining.
class ActivationFunctor
{
public:
  explicit ActivationFunctor(TfLiteFusedActivation act) : act_(act) {}

  float operator()(float a) const
  {
    switch (act_)
    {
      case kTfLiteActNone:
        return a;
      case kTfLiteActRelu:
        return a < 0.f ? 0.f : a;
      case kTfLiteActRelu6:
        return std::max(0.f, std::min(a, 6.f));
      case kTfLiteActTanh:
        return std::tanh(a);
      case kTfLiteActSigmoid:
        return 1.0f / (1.0f + std::exp(-a));
      default:
        assert(false && "Activation functor is not supported");
    }
  }

private:
  TfLiteFusedActivation act_;
};

inline void ApplyActivationToVector(const float *vector, int v_size,
                                    TfLiteFusedActivation activation, float *result)
{
  auto activation_func = ActivationFunctor(activation);
  for (int v = 0; v < v_size; v++)
  {
    *result++ = (activation_func)(*vector++);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_APPLY_ACTIVATION_TO_VECTOR_H
