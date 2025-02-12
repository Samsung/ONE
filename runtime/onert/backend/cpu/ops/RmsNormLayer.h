/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in riting, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_CPU_OPS_RMSNORM_LAYER_H__
#define __ONERT_BACKEND_CPU_OPS_RMSNORM_LAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"
#include <exec/IFunction.h>

namespace onert::backend::cpu::ops
{
class RmsNormLayer : public ::onert::exec::IFunction
{
public:
  RmsNormLayer() : _input(nullptr), _output(nullptr), _gamma(nullptr), _epsilon(1e-06f)
  {
    // Nothing
  }

public:
  void configure(const IPortableTensor *input, const IPortableTensor *gamma, float epsilon,
                 IPortableTensor *output);

  void run() override;

private:
  const IPortableTensor *_input;
  IPortableTensor *_output;
  const IPortableTensor *_gamma;
  float _epsilon;
};

} // namespace onert::backend::cpu::ops

#endif // __ONERT_BACKEND_CPU_OPS_RMSNORM_LAYER_H__
