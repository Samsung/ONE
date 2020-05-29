/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_CIRCLENODEDECL_H__
#define __LUCI_IR_CIRCLENODEDECL_H__

#include <loco/IR/Dialect.h>
#include <loco/IR/Node.h>
#include <loco/IR/NodeMixins.h>
#include <luci/IR/PropertyShapeStatus.h>

#include "CircleOpcode.h"
#include "CircleNodeVisitor.forward.h"
#include "CircleQuantParam.h"

#include <memory>

namespace luci
{

using NodeName = std::string;

struct CircleNode : public loco::Node,
                    public loco::NodeMixin<loco::NodeTrait::DataType>,
                    public loco::NodeMixin<loco::NodeTrait::TensorShape>
{
  virtual ~CircleNode() = default;

  const loco::Dialect *dialect(void) const final;
  virtual CircleOpcode opcode(void) const = 0;

  template <typename T> T accept(CircleNodeVisitorBase<T> *) const;
  template <typename T> T accept(CircleNodeMutableVisitorBase<T> *);

  NodeName name(void) const { return _name; }
  void name(const NodeName &name) { _name = name; }

  CircleQuantParam *quantparam(void) const { return _quantparam.get(); }
  void quantparam(std::unique_ptr<CircleQuantParam> &&quantparam)
  {
    _quantparam = std::move(quantparam);
  }

  bool no_shape(void) const { return _no_shape; }
  void no_shape(bool ns) { _no_shape = ns; }

  ShapeStatus shape_status(void) const { return _shape_status; }
  void shape_status(ShapeStatus ss) { _shape_status = ss; }

private:
  NodeName _name;
  std::unique_ptr<CircleQuantParam> _quantparam;
  /// @brief _no_shape is true if tensor has no shape
  bool _no_shape{false};
  ShapeStatus _shape_status{ShapeStatus::UNDEFINED};
};

template <CircleOpcode Code> struct CircleNodeImpl : public CircleNode
{
  virtual ~CircleNodeImpl() = default;

  uint32_t opnum(void) const final { return static_cast<uint32_t>(Code); }
  CircleOpcode opcode(void) const final { return Code; }
};

} // namespace luci

#endif // __LUCI_IR_CIRCLENODEDECL_H__
