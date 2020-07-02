/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_OPS_CONCATLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_CONCATLAYER_H__

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

class ConcatLayer : public ::onert::exec::IFunction
{
public:
  ConcatLayer();

public:
  template <typename T> void concatenationGeneral();

  void concatenationQuant8();

  void configure(const std::vector<const IPortableTensor *> &inputs, int32_t axis,
                 IPortableTensor *output);

  void run();

private:
  std::vector<const IPortableTensor *> _inputs;
  IPortableTensor *_output;
  int32_t _axis;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_CONCATLAYER_H__
