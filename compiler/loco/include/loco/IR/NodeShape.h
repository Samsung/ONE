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

#ifndef __LOCO_IR_NODE_SHAPE_H__
#define __LOCO_IR_NODE_SHAPE_H__

#include "loco/IR/Domain.h"

#include "loco/IR/BiasShape.h"
#include "loco/IR/DepthwiseFilterShape.h"
#include "loco/IR/FeatureShape.h"
#include "loco/IR/FilterShape.h"
#include "loco/IR/MatrixShape.h"
#include "loco/IR/TensorShape.h"

#include <vector>

namespace loco
{

class NodeShape final
{
public:
  NodeShape() = default;

public:
  NodeShape(const BiasShape &shape) { set(shape); }
  NodeShape(const DepthwiseFilterShape &shape) { set(shape); }
  NodeShape(const FeatureShape &shape) { set(shape); }
  NodeShape(const FilterShape &shape) { set(shape); }
  NodeShape(const MatrixShape &shape) { set(shape); }
  NodeShape(const TensorShape &shape) { set(shape); }

public:
  const Domain &domain(void) const { return _domain; }

public:
  void set(const BiasShape &);
  void set(const DepthwiseFilterShape &);
  void set(const FeatureShape &);
  void set(const FilterShape &);
  void set(const MatrixShape &);
  void set(const TensorShape &);

public:
  template <typename ShapeType> ShapeType as(void) const;

private:
  Domain _domain = Domain::Unknown;
  std::vector<Dimension> _dims;
};

bool operator==(const NodeShape &lhs, const NodeShape &rhs);

} // namespace loco

#endif // __LOCO_IR_NODE_SHAPE_H__
