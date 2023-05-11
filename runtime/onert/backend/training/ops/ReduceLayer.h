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

#ifndef __ONERT_BACKEND_TRAINING_OPS_REDUCESUMLAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_REDUCESUMLAYER_H__

#include "cker/neon/neon_check.h"

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>
#include <memory>

namespace nnfw
{
namespace cker
{
class Reduce;
}
} // namespace nnfw

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

enum class ReduceType
{
  kSum,
  kProd,
  kMax,
  kMin,
  kAny,
  kAll,
  kInvalid // For debug and initialize
};

class ReduceLayer : public ::onert::exec::IFunction
{
public:
  ReduceLayer();
  ~ReduceLayer();

public:
  void configure(const IPortableTensor *input, const IPortableTensor *axes, IPortableTensor *output,
                 ReduceType reduceType, bool keep_dims);

  void run() override;

private:
  const IPortableTensor *_input;
  const IPortableTensor *_axes;
  IPortableTensor *_output;

  std::unique_ptr<nnfw::cker::Reduce> _reduce_kernel;
  std::function<void(const IPortableTensor *input, IPortableTensor *output,
                     const std::vector<int> &axes)>
    _kernel;

  ReduceType _reduceType;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_REDUCESUMLAYER_H__
