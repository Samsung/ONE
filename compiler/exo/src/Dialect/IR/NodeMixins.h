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

#ifndef __DIALECT_IR_NODEMIXINS_H__
#define __DIALECT_IR_NODEMIXINS_H__

#include <loco/IR/Node.h>

namespace locoex
{

/**
 * @brief Nodes with the fixed number of inputs
 *
 * TODO Deprecated this class, and use loco::FixedArity instead
 */
template <unsigned N, typename Base> class FixedArityNode : public Base
{
public:
  FixedArityNode()
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _args[n] = std::unique_ptr<loco::Use>(new loco::Use{this});
    }
  }

  virtual ~FixedArityNode() = default;

public:
  unsigned arity(void) const final { return N; }

  loco::Node *arg(uint32_t n) const final { return _args.at(n)->node(); }

  void drop(void) final
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _args.at(n)->node(nullptr);
    }
  }

protected:
  // This API allows inherited classes to access "_args" field.
  loco::Use *at(unsigned n) const { return _args.at(n).get(); }

private:
  std::array<std::unique_ptr<loco::Use>, N> _args;
};

} // namespace locoex

#endif // __DIALECT_IR_NODEMIXINS_H__
