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

#ifndef __LOCO_IR_NODE_MIXINS_H__
#define __LOCO_IR_NODE_MIXINS_H__

#include "loco/IR/Node.h"
#include "loco/IR/DataType.h"
#include "loco/IR/Dimension.h"

#include <vector>
#include <initializer_list>

namespace loco
{

enum class NodeTrait
{
  DataType,
  // Nodes with TensorShape trait will provide the following methods:
  // - rank()
  // - rank(value)
  // - dim()
  // - dim(value)
  // - shape({...})
  TensorShape,
};

template <NodeTrait T> class NodeMixin;

template <> class NodeMixin<NodeTrait::DataType>
{
public:
  NodeMixin() = default;

public:
  const DataType &dtype(void) const { return _dtype; }
  void dtype(const DataType &dtype) { _dtype = dtype; }

private:
  /// @brief Data type
  DataType _dtype{DataType::Unknown};
};

template <> class NodeMixin<NodeTrait::TensorShape>
{
public:
  NodeMixin() = default;

public:
  uint32_t rank(void) const { return _dims.size(); }
  void rank(uint32_t value) { _dims.resize(value); }

  const Dimension &dim(uint32_t axis) const { return _dims.at(axis); }
  Dimension &dim(uint32_t axis) { return _dims.at(axis); }

  void shape(std::initializer_list<uint32_t> dims)
  {
    rank(dims.size());

    uint32_t axis = 0;
    for (auto d : dims)
    {
      dim(axis++) = d;
    }
  }

private:
  /// @brief Data shape (as tensor)
  std::vector<Dimension> _dims;
};

template <uint32_t N> struct FixedArity
{
  template <typename Base> class Mixin : public virtual Base
  {
  public:
    Mixin()
    {
      for (uint32_t n = 0; n < N; ++n)
      {
        _args[n] = std::unique_ptr<Use>{new Use{this}};
      }
    }

    virtual ~Mixin() = default;

  public:
    uint32_t arity(void) const final { return N; }

    Node *arg(uint32_t n) const final { return _args.at(n)->node(); }

    void drop(void) final
    {
      for (uint32_t n = 0; n < N; ++n)
      {
        _args.at(n)->node(nullptr);
      }
    }

  protected:
    // This API allows inherited classes to access "_args" field.
    Use *at(uint32_t n) const { return _args.at(n).get(); }

  private:
    std::array<std::unique_ptr<Use>, N> _args{};
  };
};

template <NodeTrait Trait> struct With
{
  template <typename Base> struct Mixin : public virtual Base, public NodeMixin<Trait>
  {
    // DO NOTHING
  };
};

} // namespace loco

#endif // __LOCO_IR_NODE_MIXINS_H__
