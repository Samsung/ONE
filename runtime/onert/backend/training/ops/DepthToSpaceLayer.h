/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAINING_OPS_DEPTH_TO_SPACE_LAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_DEPTH_TO_SPACE_LAYER_H__

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
class DepthToSpaceLayer : public ::onert::exec::IFunction
{
public:
  DepthToSpaceLayer();

  void configure(const IPortableTensor *input, const int32_t block_size, IPortableTensor *output);

  void run() override;

private:
  template <typename T> void depthToSpace();

  const IPortableTensor *_input;
  int32_t _block_size;
  IPortableTensor *_output;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_DEPTH_TO_SPACE_LAYER_H__
