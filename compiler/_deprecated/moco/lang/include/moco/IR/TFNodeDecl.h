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

#ifndef __MOCO_IR_TFNODE_DECL_H__
#define __MOCO_IR_TFNODE_DECL_H__

#include <loco/IR/Node.h>
#include <loco/IR/Dialect.h>

#include "moco/IR/TFOpcode.h"
#include "moco/IR/TFNodeVisitor.forward.h"

#include "moco/IR/TFDataLayout.h"
#include "moco/IR/TFPadding.h"

#include <array>
#include <string>

namespace moco
{

/**
 * @note  NodeName is string name of the Node without ':#' prefix like ':0' or ':1'
 */
using NodeName = std::string;

struct TFNode : public loco::Node
{
  virtual ~TFNode() = default;

  const loco::Dialect *dialect(void) const final;
  virtual TFOpcode opcode(void) const = 0;

  template <typename T> T accept(TFNodeVisitorBase<T> *) const;
  template <typename T> T accept(TFNodeMutableVisitorBase<T> *);

  NodeName name(void) const { return _name; }
  void name(const NodeName &name) { _name = name; }

private:
  NodeName _name;
};

template <TFOpcode Code> struct TFNodeImpl : public TFNode
{
  virtual ~TFNodeImpl() = default;

  uint32_t opnum(void) const final { return static_cast<uint32_t>(Code); }
  TFOpcode opcode(void) const final { return Code; }
};

/**
 * @brief Nodes with the fixed number of inputs
 */
template <unsigned N, typename Base> class FixedArityNode : public Base
{
public:
  FixedArityNode()
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _args[n] = std::unique_ptr<loco::Use>{new loco::Use{this}};
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
  std::array<std::unique_ptr<loco::Use>, N> _args{};
};

} // namespace moco

#endif // __MOCO_IR_TFNODE_DECL_H__
