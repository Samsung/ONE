/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExpressionCache.h"

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

// Check common (non-op-specific) attributes of lhs and rhs
bool same_common_attributes(const luci::CircleNode *lhs, const luci::CircleNode *rhs)
{
  // Opcode
  if (lhs->opcode() != rhs->opcode())
    return false;

  // Shape
  if (lhs->rank() != rhs->rank())
    return false;

  for (uint32_t i = 0; i < lhs->rank(); i++)
  {
    if (lhs->dim(i).known() != rhs->dim(i).known())
      return false;

    if (lhs->dim(i).value() != rhs->dim(i).value())
      return false;
  }

  // Data type
  if (lhs->dtype() != rhs->dtype())
    return false;

  // Op version
  if (lhs->op_version() != rhs->op_version())
    return false;

  // QuantParam
  const auto lhs_qparam = lhs->quantparam();
  const auto rhs_qparam = rhs->quantparam();

  if (lhs_qparam == nullptr and rhs_qparam != nullptr)
    return false;

  if (lhs_qparam != nullptr and rhs_qparam == nullptr)
    return false;

  if (lhs_qparam)
  {
    assert(rhs_qparam); // FIX_ME_UNLESS

    if (lhs_qparam->scale != rhs_qparam->scale)
      return false;

    if (lhs_qparam->zerop != rhs_qparam->zerop)
      return false;

    if (lhs_qparam->min != rhs_qparam->min)
      return false;

    if (lhs_qparam->max != rhs_qparam->max)
      return false;
  }

  return true;
}

// Return true if two constants are the same
bool same_const(const luci::CircleConst *x, const luci::CircleConst *y)
{
  assert(x != nullptr); // FIX_CALLER_UNLESS
  assert(y != nullptr); // FIX_CALLER_UNLESS

  RETURN_FALSE_UNLESS(same_common_attributes(x, y));

  switch (x->dtype())
  {
    case loco::DataType::S32:
    {
      const auto size_x = x->size<loco::DataType::S32>();
      const auto size_y = y->size<loco::DataType::S32>();
      RETURN_FALSE_UNLESS(size_x == size_y);

      for (uint32_t i = 0; i < size_x; i++)
      {
        RETURN_FALSE_UNLESS(x->at<loco::DataType::S32>(i) == y->at<loco::DataType::S32>(i));
      }
      return true;
    }
    // TODO Support more dtypes
    default:
      // Simply return false
      return false;
  }

  return true;
}

// Return true if x and y are semantically equal
bool same_attributes(const luci::CircleTranspose *x, luci::CircleTranspose *y)
{
  assert(x != nullptr); // FIX_CALLER_UNLESS
  assert(y != nullptr); // FIX_CALLER_UNLESS

  assert(same_common_attributes(x, y)); // FIX_CALLER_UNLESS

  const auto perm_x = dynamic_cast<luci::CircleConst *>(x->perm());
  const auto perm_y = dynamic_cast<luci::CircleConst *>(y->perm());

  RETURN_FALSE_UNLESS(perm_x);
  RETURN_FALSE_UNLESS(perm_y);

  // Check perm_x and perm_y are the same
  RETURN_FALSE_UNLESS(same_const(perm_x, perm_y));

  return true;
}

// Use a similar approach as boost's hash_combine
template <class T> inline void hash_combine(std::size_t &s, const T v)
{
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

template <> inline void hash_combine(std::size_t &s, luci::CircleNode *node)
{
  // Shape
  hash_combine(s, node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    hash_combine(s, node->dim(i).value());

  // Data type
  hash_combine(s, static_cast<std::size_t>(node->dtype()));

  // Op version
  hash_combine(s, node->op_version());

  // Op code
  hash_combine(s, node->opcode());

  // QuantParam
  // Let's skip quantparam to reduce burden of hash function
}

} // namespace

namespace luci
{
namespace pass
{

Expression Expression::build(luci::CircleNode *node)
{
  if (node == nullptr)
    throw std::invalid_argument("node");

  Expression key;
  {
    switch (node->opcode())
    {
      case luci::CircleOpcode::QUANTIZE:
      case luci::CircleOpcode::TRANSPOSE:
        key.inputs.emplace_back(node->arg(0));
        break;
      // TODO Add more Ops
      default:
        // Return invalid expression
        key.op = nullptr;
        return key;
    }

    key.op = node;
  }

  return key;
}

bool operator==(const Expression &x, const Expression &y)
{
  if (x.inputs != y.inputs)
    return false;

  // Check general (non-op-specific) attributes
  if (not same_common_attributes(x.op, y.op))
    return false;

  assert(x.op->opcode() == y.op->opcode()); // FIX_ME_UNLESS

  // Check op-specific attributes
  switch (x.op->opcode())
  {
    case luci::CircleOpcode::QUANTIZE:
    {
      // This Op has no op-specific attribute.
      // same_common_attributes is enough.
      return true;
    }
    case luci::CircleOpcode::TRANSPOSE:
    {
      const auto trans_x = loco::must_cast<luci::CircleTranspose *>(x.op);
      const auto trans_y = loco::must_cast<luci::CircleTranspose *>(y.op);

      return same_attributes(trans_x, trans_y);
    }
    // TODO Implement more operators
    default:
      // NYI: Unsupported operators
      return false;
  }

  return true;
}

std::size_t Expression::Hash::call(const Expression &k) const noexcept
{
  std::size_t res = 0;
  for (const auto input : k.inputs)
    hash_combine(res, input);

  hash_combine(res, k.op);

  return res;
}

} // namespace pass
} // namespace luci
