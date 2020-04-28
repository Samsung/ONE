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

#include "Dropout.h"

#include "AttributeHelpers.h"

namespace mir_onnx
{

void convertDropoutV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // consumed_inputs attribute not used
  convertDropoutV6(onnx_node, context);
}

void convertDropoutV6(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto is_test = getAttributeValue<std::int64_t>(onnx_node, "is_test", 0);
  if (is_test == 0)
    throw std::runtime_error("Not supported is_test attribute!");

  convertDropoutV10(onnx_node, context);
}

void convertDropoutV7(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertDropoutV10(onnx_node, context);
}

void convertDropoutV10(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);

  // ratio attribute not used

  // This is a no-op in inference mode.
  context->setNodeOutputs(onnx_node, {inputs[0]});
}

} // namespace mir_onnx
