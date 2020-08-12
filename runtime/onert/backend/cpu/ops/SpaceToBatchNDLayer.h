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

#ifndef __ONERT_BACKEND_CPU_OPS_SPACE_TO_BATCH_ND_LAYER_H__
#define __ONERT_BACKEND_CPU_OPS_SPACE_TO_BATCH_ND_LAYER_H__

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
class SpaceToBatchNDLayer : public ::onert::exec::IFunction
{
public:
  SpaceToBatchNDLayer();

  void configure(const IPortableTensor *input, const IPortableTensor *block_shape,
                 const IPortableTensor *padding, IPortableTensor *output);

  void run() override;

  const backend::ITensor *getOutput(int output_ind = 0) const override
  {
    assert(output_ind == 0);
    return _output;
  }

private:
  void checkDimension();

  template <typename T> uint32_t getPad();

  template <typename T> void spaceToBatchND();

  const IPortableTensor *_input;
  const IPortableTensor *_block_shape;
  const IPortableTensor *_padding;
  IPortableTensor *_output;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_SPACE_TO_BATCH_ND_LAYER_H__
