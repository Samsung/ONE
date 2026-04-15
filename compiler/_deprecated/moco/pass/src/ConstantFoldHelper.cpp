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

#include "ConstantFoldHelper.h"

#include <cassert>
#include <sstream>
#include <string>

namespace
{

// TODO this may need to be moved to loco
bool same_shape(const loco::TensorShape *lhs, const loco::TensorShape *rhs)
{
  if (lhs->rank() != rhs->rank())
    return false;

  for (uint32_t r = 0; r < lhs->rank(); r++)
  {
    if (lhs->dim(r).value() != rhs->dim(r).value())
      return false;
  }
  return true;
}

} // namespace

namespace moco
{

TFConst *new_const(loco::Graph *graph, loco::TensorShape &tensor_shape, const loco::DataType &dtype)
{
  assert(dtype == loco::DataType::S32 || dtype == loco::DataType::FLOAT32);

  auto const_node = graph->nodes()->create<TFConst>();
  const_node->dtype(dtype);
  const_node->rank(tensor_shape.rank());

  // Calc number of elements for target node and set shape
  uint32_t num_elements = 1;
  for (uint32_t r = 0; r < tensor_shape.rank(); r++)
  {
    const_node->dim(r) = tensor_shape.dim(r);
    assert(const_node->dim(r).known());
    num_elements = num_elements * const_node->dim(r).value();
  }
  if (dtype == loco::DataType::S32)
    const_node->size<loco::DataType::S32>(num_elements);
  else if (dtype == loco::DataType::FLOAT32)
    const_node->size<loco::DataType::FLOAT32>(num_elements);

  // give name for this node from address to be unique
  std::ostringstream oss;
  oss << "Const_" << (void *)const_node;
  const_node->name(oss.str());

  return const_node;
}

} // namespace moco

namespace moco
{

template <> int32_t scalar_from_const<int32_t>(const TFConst *tfconst)
{
  assert(tfconst->rank() == 0 || tfconst->rank() == 1);
  assert(tfconst->dtype() == loco::DataType::S32);
  return tfconst->at<loco::DataType::S32>(0);
}

template <> float scalar_from_const<float>(const TFConst *tfconst)
{
  assert(tfconst->rank() == 0 || tfconst->rank() == 1);
  assert(tfconst->dtype() == loco::DataType::FLOAT32);
  return tfconst->at<loco::DataType::FLOAT32>(0);
}

bool valid_shape_for_constfold_binary_op(const loco::TensorShape &lhs, const loco::TensorShape &rhs)
{
  // scalar
  if (lhs.rank() == 0 || rhs.rank() == 0)
    return true;

  // same as scalar
  if (lhs.rank() == 1 && lhs.dim(0).value() == 1)
    return true;
  if (rhs.rank() == 1 && rhs.dim(0).value() == 1)
    return true;

  // for elementwise binary operation
  return ::same_shape(&lhs, &rhs);
}

} // namespace moco

namespace moco
{

float BinaryFunc::apply(float, float) const
{
  throw std::runtime_error{"F32 is not supported yet"};
}

int32_t BinaryFunc::apply(int32_t, int32_t) const
{
  throw std::runtime_error{"S32 is not supported yet"};
}

} // namespace moco

namespace
{

void apply_binary_s32(const moco::TFConst *lhs, int32_t rhs, moco::TFConst *output,
                      const moco::BinaryFunc &f)
{
  assert(lhs->dtype() == loco::DataType::S32);
  assert(same_shape(lhs, output));

  uint32_t nume = num_elements(lhs);
  for (uint32_t e = 0; e < nume; e++)
  {
    output->at<loco::DataType::S32>(e) = f.apply(lhs->at<loco::DataType::S32>(e), rhs);
  }
}

void apply_binary_f32(const moco::TFConst *lhs, float rhs, moco::TFConst *output,
                      const moco::BinaryFunc &f)
{
  assert(lhs->dtype() == loco::DataType::FLOAT32);
  assert(same_shape(lhs, output));

  uint32_t nume = num_elements(lhs);
  for (uint32_t e = 0; e < nume; e++)
  {
    output->at<loco::DataType::FLOAT32>(e) = f.apply(lhs->at<loco::DataType::FLOAT32>(e), rhs);
  }
}

void apply_binary_s32(const moco::TFConst *lhs, const moco::TFConst *rhs, moco::TFConst *output,
                      const moco::BinaryFunc &f)
{
  assert(same_shape(output, lhs));
  assert(same_shape(output, rhs));
  assert(output->dtype() == lhs->dtype());
  assert(output->dtype() == rhs->dtype());

  uint32_t nume = num_elements(lhs);
  for (uint32_t e = 0; e < nume; e++)
  {
    output->at<loco::DataType::S32>(e) =
      f.apply(lhs->at<loco::DataType::S32>(e), rhs->at<loco::DataType::S32>(e));
  }
}

void apply_binary_f32(const moco::TFConst *lhs, const moco::TFConst *rhs, moco::TFConst *output,
                      const moco::BinaryFunc &f)
{
  assert(same_shape(output, lhs));
  assert(same_shape(output, rhs));
  assert(output->dtype() == lhs->dtype());
  assert(output->dtype() == rhs->dtype());

  uint32_t nume = num_elements(lhs);
  for (uint32_t e = 0; e < nume; e++)
  {
    output->at<loco::DataType::FLOAT32>(e) =
      f.apply(lhs->at<loco::DataType::FLOAT32>(e), rhs->at<loco::DataType::FLOAT32>(e));
  }
}

} // namespace

namespace moco
{

template <>
void apply_binary<int32_t>(const moco::TFConst *x_const, const moco::TFConst *y_const,
                           moco::TFConst *output_const, const moco::BinaryFunc &f)
{
  auto x_shape = moco::tensor_shape(x_const);
  auto y_shape = moco::tensor_shape(y_const);

  if (y_shape.rank() == 0 || y_shape.rank() == 1)
  {
    auto rhs = scalar_from_const<int32_t>(y_const);
    apply_binary_s32(x_const, rhs, output_const, f);
  }
  else if (x_shape.rank() == 0 || x_shape.rank() == 1)
  {
    auto rhs = scalar_from_const<int32_t>(x_const);
    apply_binary_s32(y_const, rhs, output_const, f);
  }
  else
  {
    apply_binary_f32(x_const, y_const, output_const, f);
  }
}

template <>
void apply_binary<float>(const moco::TFConst *x_const, const moco::TFConst *y_const,
                         moco::TFConst *output_const, const moco::BinaryFunc &f)
{
  auto x_shape = moco::tensor_shape(x_const);
  auto y_shape = moco::tensor_shape(y_const);

  if (y_shape.rank() == 0 || y_shape.rank() == 1)
  {
    auto rhs = scalar_from_const<float>(y_const);
    apply_binary_f32(x_const, rhs, output_const, f);
  }
  else if (x_shape.rank() == 0 || x_shape.rank() == 1)
  {
    auto rhs = scalar_from_const<float>(x_const);
    apply_binary_f32(y_const, rhs, output_const, f);
  }
  else
  {
    apply_binary_f32(x_const, y_const, output_const, f);
  }
}

} // namespace moco
