/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_INTERPRETER_IMPORT_OP_CIRCLE_REFERENCING_CONST_H__
#define __LUCI_INTERPRETER_IMPORT_OP_CIRCLE_REFERENCING_CONST_H__

#include <luci/Import/NodeBuilder.h>

#include <luci/IR/Nodes/CircleConst.h>

namespace luci_interpreter
{
using namespace luci;

/**
 * @brief Builder creates CircleCustom node with pointer to constants data from Tensor with buffer.
 */
class CircleReferencingConstNodeBuilder : public TypedNodeBuilder<NodeBuilderType::BUFFER>
{
public:
  CircleNode *build(TensorIndex tensor_index, GraphBuilderContext *ctx) const final;
};

} // namespace luci_interpreter

#endif // __LUCI_INTERPRETER_IMPORT_OP_CIRCLE_REFERENCING_CONST_H__
