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

#ifndef __LUCI_IR_VARIADICARITYNODES_H__
#define __LUCI_IR_VARIADICARITYNODES_H__

#include <loco/IR/Node.h>
#include <loco/IR/Use.h>

#include <vector>
#include <memory>
#include <cassert>

namespace luci
{

/**
 * @brief Nodes with the variadic inputs
 */
template <typename Base> class VariadicArityNode : public Base
{
public:
  VariadicArityNode(uint32_t arity)
  {
    for (uint32_t n = 0; n < arity; ++n)
    {
      _args.push_back(std::make_unique<loco::Use>(this));
    }
  };

  virtual ~VariadicArityNode() = default;

public:
  uint32_t arity(void) const final { return _args.size(); }

  loco::Node *arg(uint32_t n) const final { return _args.at(n)->node(); }

  void drop(void) final
  {
    for (uint32_t n = 0; n < _args.size(); ++n)
    {
      _args.at(n)->node(nullptr);
    }
  }

protected:
  // This API allows inherited classes to access "_args" field.
  loco::Use *at(uint32_t n) const { return _args.at(n).get(); }

private:
  std::vector<std::unique_ptr<loco::Use>> _args;
};

} // namespace luci

#endif // __LUCI_IR_VARIADICARITYNODES_H__
