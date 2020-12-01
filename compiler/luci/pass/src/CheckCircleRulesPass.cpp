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

#include "luci/Pass/CheckCircleRulesPass.h"

#include <luci/IR/CircleShapeSignature.h>
#include <luci/Service/CircleShapeInference.h>
#include <luci/Service/CircleTypeInference.h>
#include <luci/Service/CircleShapeSignatureInference.h>
#include <luci/Log.h>

#include <loco.h>

namespace
{

std::ostream &operator<<(std::ostream &os, const loco::TensorShape &tensor_shape)
{
  os << "[";
  for (uint32_t r = 0; r < tensor_shape.rank(); ++r)
  {
    if (r)
      os << ",";
    os << tensor_shape.dim(r).value();
  }
  os << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const luci::CircleNode *circle_node)
{
  os << "[";
  for (uint32_t r = 0; r < circle_node->rank(); ++r)
  {
    if (r)
      os << ",";
    os << circle_node->dim(r).value();
  }
  os << "]";

  os << " (";
  for (uint32_t r = 0; r < circle_node->shape_signature().rank(); ++r)
  {
    if (r)
      os << ",";
    os << circle_node->shape_signature().dim(r);
  }
  os << ")";
  return os;
}

bool is_same_shape(luci::CircleNode *node, loco::TensorShape shape)
{
  if (node->rank() != shape.rank())
    return false;

  for (uint32_t i = 0; i < node->rank(); ++i)
    if (!(node->dim(i) == shape.dim(i)))
      return false;

  return true;
}

} // namespace

namespace luci
{

bool CheckCircleRulesPass::run(luci::Module *m)
{
  bool changed = false;

  for (size_t g = 0; g < m->size(); ++g)
  {
    if (run(m->graph(g)))
      changed = true;
  }

  return changed;
}

bool CheckCircleRulesPass::run(loco::Graph *g)
{
  LOGGER(l);

  luci::ssinf::Rule sig_infer_rule;
  luci::sinf::Rule shape_infer_rule;
  luci::tinf::Rule type_infer_rule;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    loco::TensorShape shape;
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    if (shape_infer_rule.infer(circle_node, shape) && !is_same_shape(circle_node, shape))
    {
      WARN(l) << "  loco rule : " << shape << std::endl;
      WARN(l) << "circle rule : " << circle_node << std::endl;
      assert(false && "Inferred shape is different");
    }

    loco::DataType dtype;
    if (type_infer_rule.infer(circle_node, dtype) && circle_node->dtype() != dtype)
    {
      WARN(l) << "  loco rule : " << (int)dtype << std::endl;
      WARN(l) << "circle rule : " << (int)circle_node->dtype() << std::endl;
      assert(false && "Inferred dtype is different");
    }

    luci::ShapeSignature sig;
    if (sig_infer_rule.infer(circle_node, sig))
    {
      for (uint32_t i = 0; i < sig.rank(); ++i)
      {
        if (sig.dim(i) != -1 && shape.dim(i).value() != (uint32_t)sig.dim(i))
        {
          WARN(l) << "loco shape : " << shape << std::endl;
          WARN(l) << "circle sig : " << circle_node << std::endl;
          assert(false && "Inferred signature is not compatible with shape");
        }
      }
    }
  }

  return false;
}

} // namespace luci
