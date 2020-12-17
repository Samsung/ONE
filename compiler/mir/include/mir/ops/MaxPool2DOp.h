/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_OPS_MAX_POOL_OP_H_
#define _MIR_OPS_MAX_POOL_OP_H_

#include "mir/Operation.h"
#include "mir/Attributes.h"

#include <cstdint>
#include <vector>

namespace mir
{
namespace ops
{

class MaxPool2DOp : public Operation
{
public:
  MaxPool2DOp(Output *arg, const MaxPool2DOpAttributes &attributes)
    : Operation(Type::maxPool2D, {arg}), _attributes(attributes)
  {
    inferOutputTypes();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new MaxPool2DOp(inputs[0], _attributes);
  };

  const std::vector<std::int32_t> &getWindowSize() const { return _attributes.window; }

  const std::vector<std::int32_t> &getStrides() const { return _attributes.strides; }

  const std::vector<std::int32_t> &getPaddingBefore() const { return _attributes.padding_before; }

  const std::vector<std::int32_t> &getPaddingAfter() const { return _attributes.padding_after; }

  DataFormat getDataFormat() const { return _attributes.data_format; }

  const MaxPool2DOpAttributes &getAttributes() const { return _attributes; }

private:
  void inferOutputTypes();

  MaxPool2DOpAttributes _attributes;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_MAX_POOL_OP_H_
