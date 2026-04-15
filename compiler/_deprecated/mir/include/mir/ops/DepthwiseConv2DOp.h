/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_OPS_DEPTHWISE_CONV_2D_OP_H_
#define _MIR_OPS_DEPTHWISE_CONV_2D_OP_H_

#include "mir/Operation.h"
#include "mir/Attributes.h"
#include <vector>

namespace mir
{
namespace ops
{

class DepthwiseConv2DOp : public Operation
{
public:
  DepthwiseConv2DOp(Output *input, Output *kernel, const Conv2DOpAttributes &attributes)
    : Operation(Type::depthwiseConv, {input, kernel}), _attributes(attributes)
  {
    inferOutputTypes();
  }

  DepthwiseConv2DOp(Output *input, Output *kernel, Output *bias,
                    const Conv2DOpAttributes &attributes)
    : Operation(Type::depthwiseConv, {input, kernel, bias}), _attributes(attributes)
  {
    inferOutputTypes();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    if (inputs.size() == 2)
      return new DepthwiseConv2DOp(inputs[0], inputs[1], _attributes);
    else
      return new DepthwiseConv2DOp(inputs[0], inputs[1], inputs[2], _attributes);
  }

  const std::vector<std::int32_t> &getStrides() const { return _attributes.strides; }

  const std::vector<std::int32_t> &getPaddingBefore() const { return _attributes.padding_before; }

  const std::vector<std::int32_t> &getPaddingAfter() const { return _attributes.padding_after; }

  DataFormat getDataFormat() const { return _attributes.data_format; }

  const Conv2DOpAttributes &getAttributes() const { return _attributes; }

private:
  void inferOutputTypes();

  mir::Conv2DOpAttributes _attributes;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_DEPTHWISE_CONV_2D_OP_H_
