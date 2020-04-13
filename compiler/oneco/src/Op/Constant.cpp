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

#include "Constant.h"

#include <cassert>

namespace moco
{
namespace onnx
{

bool ConstantGraphBuilder::validate(OpsetVersion opset_version, const ::onnx::NodeProto &node) const
{
  if (opset_version >= 9)
    return Constant_V9().validate(node);
  else if (opset_version >= 1)
    return Constant_V1().validate(node);
  else
    throw std::runtime_error("Invalid ONNX IR version");
}

void ConstantGraphBuilder::build(OpsetVersion opset_version, const ::onnx::NodeProto &node,
                                 GraphBuilderContext *context) const
{
  if (opset_version >= 9)
    Constant_V9().build(node, context);
  else if (opset_version >= 1)
    Constant_V1().build(node, context);
  else
    throw std::runtime_error("Invalid ONNX IR version");
}

} // namespace onnx
} // namespace moco

#include "GraphBuilderRegistry.h"

REGISTER_OP_BUILDER(Constant, ConstantGraphBuilder)
