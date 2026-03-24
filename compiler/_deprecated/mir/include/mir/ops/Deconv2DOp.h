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

#ifndef _MIR_OPS_DECONV_2D_OP_H_
#define _MIR_OPS_DECONV_2D_OP_H_

#include "mir/Operation.h"
#include "mir/Attributes.h"
#include "mir/ops/PaddingType.h"

#include <cstdint>
#include <vector>

namespace mir
{
namespace ops
{

class DeConv2DOp : public Operation
{
public:
  DeConv2DOp(Output *input, Output *kernel, const Deconv2DOpAttributes &attributes)
    : Operation(Type::deConv2D, {input, kernel}), _attributes(attributes)
  {
    inferOutputTypes();
  }

  DeConv2DOp(Output *input, Output *kernel, const Deconv2DOpAttributes &attributes,
             const Shape &output_shape)
    : Operation(Type::deConv2D, {input, kernel}), _attributes(attributes)
  {
    assert(input->getElementType() == kernel->getElementType());
    setOutputType(0, {input->getElementType(), output_shape});
    inferPaddings();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    if (getPaddingType() == PaddingType::Explicit)
      return new DeConv2DOp(inputs[0], inputs[1], _attributes);
    else
      return new DeConv2DOp(inputs[0], inputs[1], _attributes, getOutputShape(0));
  }

  const std::vector<std::int32_t> &getStrides() const { return _attributes.strides; }

  PaddingType getPaddingType() const { return _attributes.padding_type; }

  const std::vector<std::int32_t> &getPaddingBefore() const { return _attributes.padding_before; }

  const std::vector<std::int32_t> &getPaddingAfter() const { return _attributes.padding_after; }

  DataFormat getDataFormat() const { return _attributes.data_format; }

  const Deconv2DOpAttributes &getAttributes() const { return _attributes; }

private:
  void inferOutputTypes();

  /**
   * @brief Compute paddings based on input shape, kernel shape and strides
   */
  void inferPaddings();

  Deconv2DOpAttributes _attributes;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_DECONV_2D_OP_H_
