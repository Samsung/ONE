/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_EXPORT_BUILTIN_TYPES_MAPPING_RULE_H__
#define __CIRCLE_EXPORT_BUILTIN_TYPES_MAPPING_RULE_H__

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

class BuiltinOperatorMappingRule final : public CircleNodeVisitor<circle::BuiltinOperator>
{
public:
  BuiltinOperatorMappingRule()
  {
    // DO NOTHING
  }

public:
  static BuiltinOperatorMappingRule &get()
  {
    static BuiltinOperatorMappingRule instance;
    return instance;
  }

public:
#define CIRCLE_NODE(CIRCLE_NODE, OP, OPTION) \
  circle::BuiltinOperator visit(const CIRCLE_NODE *) final { return circle::OP; }
// Virtual nodes are not circle builtin operator
#define CIRCLE_VNODE(CIRCLE_NODE)
#include "CircleOps.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE
};

class BuiltinOptionsMappingRule final : public CircleNodeVisitor<circle::BuiltinOptions>
{
public:
  BuiltinOptionsMappingRule()
  {
    // DO NOTHING
  }

public:
  static BuiltinOptionsMappingRule &get()
  {
    static BuiltinOptionsMappingRule instance;
    return instance;
  }

public:
#define CIRCLE_NODE(CIRCLE_NODE, OP, OPTION) \
  circle::BuiltinOptions visit(const CIRCLE_NODE *) final { return circle::OPTION; }
// Virtual nodes are not circle builtin operator
#define CIRCLE_VNODE(CIRCLE_NODE)
#include "CircleOps.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE
};

} // namespace luci

#endif // __CIRCLE_EXPORT_BUILTIN_TYPES_MAPPING_RULE_H__
