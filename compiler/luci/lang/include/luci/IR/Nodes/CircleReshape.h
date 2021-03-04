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

#ifndef __LUCI_IR_CIRCLERESHAPE_H__
#define __LUCI_IR_CIRCLERESHAPE_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief RESHAPE in Circle
 */
class CircleReshape final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::RESHAPE>>
{
public:
  CircleReshape() = default;

public:
  loco::Node *tensor(void) const { return at(0)->node(); }
  void tensor(loco::Node *node) { at(0)->node(node); }

  // NOTE shape is optional and can be CircleConst or any other type
  //      and also should be CircleOutputDummy when reshape option does not exist
  loco::Node *shape(void) const { return at(1)->node(); }
  void shape(loco::Node *node) { at(1)->node(node); }

public:
  class Shape
  {
  public:
    uint32_t rank(void) const { return _shape.size(); }
    void rank(uint32_t rank) { _shape.resize(rank); }

    int32_t dim(uint32_t n) const { return _shape.at(n); }
    int32_t &dim(uint32_t n) { return _shape.at(n); }

  private:
    std::vector<int32_t> _shape;
  };

  const Shape *newShape(void) const { return &_new_shape; }
  Shape *newShape(void) { return &_new_shape; }

private:
  Shape _new_shape;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLERESHAPE_H__
