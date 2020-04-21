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

#include <mio/circle/schema_generated.h>
#include <cassert>

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
  const std::vector<CircleNode *> &inputs(void) const { return _inputs; }
  void inputs(const std::vector<CircleNode *> &inputs) { _inputs = std::move(inputs); }

  const circle::BuiltinOptionsUnion &builtin_options(void) const { return _builtin_options; }
  void builtin_options(const circle::BuiltinOptionsUnion &builtin_options)
  {
    _builtin_options = builtin_options;
  }

  const std::vector<uint8_t> &custom_options(void) const { return _custom_options; }
  void custom_options(const std::vector<uint8_t> &custom_options)
  {
    _custom_options = std::move(custom_options);
  }

  const std::string &custom_code(void) const { return _custom_code; }
  void custom_code(const std::string &custom_code) { _custom_code = custom_code; }

private:
  std::vector<CircleNode *> _inputs;
  circle::BuiltinOptionsUnion _builtin_options;
  std::vector<uint8_t> _custom_options;
  std::string _custom_code;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLECUSTOM_H__
