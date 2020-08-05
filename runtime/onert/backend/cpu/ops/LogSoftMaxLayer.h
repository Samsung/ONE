/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_OPS_LOGSOFTMAXLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_LOGSOFTMAXLAYER_H__

#include "../Tensor.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

class LogSoftMaxLayer : public ::onert::exec::IFunction
{
public:
  LogSoftMaxLayer();

public:
  void logsoftmaxFloat32();

  void logsoftmaxQuant8();

  void configure(const IPortableTensor *input, const float beta, const int axis,
                 IPortableTensor *output);

  void run();

private:
  const IPortableTensor *_input;
  IPortableTensor *_output;

  float _beta;
  int _axis;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_LOGSOFTMAXLAYER_H__
