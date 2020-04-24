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

#ifndef __LUCI_IR_CIRCLECUSTOM_H__
#define __LUCI_IR_CIRCLECUSTOM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/VariadicArityNode.h"

namespace luci
{

/**
 * @brief CUSTOM in Circle
 */
class CircleCustom final : public VariadicArityNode<CircleNodeImpl<CircleOpcode::CUSTOM>>
{
public:
  CircleCustom(uint32_t arity) : VariadicArityNode<CircleNodeImpl<CircleOpcode::CUSTOM>>(arity)
  {
    // TODO Support when arity is 0
    assert(arity >= 1);
  }

public:
  uint32_t numInputs(void) const { return arity(); }

public:
  Node *inputs(uint32_t index) const { return at(index)->node(); }
  void inputs(uint32_t index, Node *node) { at(index)->node(node); }

  const std::vector<uint8_t> &custom_options(void) const { return _custom_options; }
  void custom_options(const std::vector<uint8_t> &custom_options)
  {
    _custom_options = custom_options;
  }

  const std::string &custom_code(void) const { return _custom_code; }
  void custom_code(const std::string &custom_code) { _custom_code = custom_code; }

private:
  std::vector<uint8_t> _custom_options;
  std::string _custom_code;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLECUSTOM_H__
