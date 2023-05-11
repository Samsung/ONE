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

#ifndef __ONERT_BACKEND_TRAINING_OPS_ONEHOTLAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_ONEHOTLAYER_H__

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

class OneHotLayer : public ::onert::exec::IFunction
{
public:
  OneHotLayer()
    : _indices(nullptr), _depth(nullptr), _on_value(nullptr), _off_value(nullptr), _output(nullptr),
      _axis(-1)
  {
    // DO NOTHING
  }

public:
  template <typename T> void oneHotImpl();

  void configure(const IPortableTensor *indices, const IPortableTensor *depth,
                 const IPortableTensor *on_value, const IPortableTensor *off_value,
                 IPortableTensor *output, int32_t axis);

  void run() override;

private:
  const IPortableTensor *_indices;
  const IPortableTensor *_depth;
  const IPortableTensor *_on_value;
  const IPortableTensor *_off_value;
  IPortableTensor *_output;

  int32_t _axis;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_ONEHOTLAYER_H__
