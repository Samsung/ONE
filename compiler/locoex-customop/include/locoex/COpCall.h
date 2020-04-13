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

#ifndef __LOCOEX_COPCALL_H__
#define __LOCOEX_COPCALL_H__

#include "VariadicArityNode.h"
#include "locoex/COpAttrTypes.h"
#include "locoex/COpNode.h"

#include <loco/IR/NodeMixins.h>

#include <map>
#include <memory>

namespace locoex
{

/**
 * @brief Class to calls custom operation
 */
class COpCall final : public VariadicArityNode<COpNode>,
                      public loco::NodeMixin<loco::NodeTrait::TensorShape>,
                      public loco::NodeMixin<loco::NodeTrait::DataType>
{
public:
  COpCall(unsigned arity) : VariadicArityNode<COpNode>(arity) {}

public:
  void op(const std::string &op) { _op.assign(op); }
  const std::string &op() { return _op; }

  void name(const std::string &name) { _name.assign(name); }
  const std::string &name() { return _name; }

  void input(uint32_t nth, loco::Node *node) { at(nth)->node(node); }
  loco::Node *input(uint32_t nth) const { return at(nth)->node(); }

  /// @brief  Store [attr_name, attr_data]
  void attr(const std::string &attr_name, std::unique_ptr<COpAttrData> &&attr_data);

  /// @brief  Retrieve attr_data stored with attr_name
  template <COpAttrType AT>
  const typename AttrTypeTrait<AT>::Type *attr(const std::string &attr_name) const;

  /// @brief get all the names of attr
  std::vector<std::string> attr_names() const;

private:
  std::string _op;
  std::string _name;

  std::map<std::string, std::unique_ptr<COpAttrData>> _attrs;
};

} // namespace locoex

#endif // __LOCOEX_COPCALL_H__
