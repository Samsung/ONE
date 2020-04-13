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

#ifndef __TF_REDUCE_CANONICALIZE_HELPER_H__
#define __TF_REDUCE_CANONICALIZE_HELPER_H__

#include <moco/IR/TFDialect.h>
#include <moco/IR/TFNodes.h>

#include <loco/Service/ShapeInference.h>

#include <moco/Log.h>

namespace
{

template <typename TFNodeT> loco::ReduceFunc reduceFunc(void);

template <> loco::ReduceFunc reduceFunc<moco::TFMean>(void) { return loco::ReduceFunc::Mean; }

template <typename TFNode> bool canonicalize_reduce_node(TFNode *node)
{
  LOGGER(l);

  INFO(l) << "TFNodeCanonicalize ReduceNode begin";

  auto graph = node->graph();

  /**
   * This will replace T/F Reduce node with a corresponding Canonical Reduce node
   *
   * BEFORE
   *   reduction_indices -------- T/F Node -- C
   *               input -------/
   *
   * AFTER
   *                      +------ T/F Node --
   *                      |     /
   *   reduction_indices -------
   *                      |     \
   *               input -+------ Canonical Node -- C
   *
   * NOTE
   *   - T/F Node is disconnected from C after transformation
   */

  // TFSqueeze had to be inserted if keep_dims() was false
  assert(node->keep_dims());

  auto axes_node = node->reduction_indices();
  assert(axes_node != nullptr);

  auto node_tensor_shape = loco::shape_get(node).template as<loco::TensorShape>();

  // Canonicalization into TensorReduce is valid when reduction indices is constant
  // TODO Support general TensorReduce case
  std::vector<int32_t> axes_values;
  if (auto const_axes = dynamic_cast<moco::TFConst *>(axes_node))
  {
    // TODO Support S64 type
    assert(const_axes->dtype() == loco::DataType::S32);

    for (uint32_t i = 0; i < const_axes->size<loco::DataType::S32>(); ++i)
    {
      int32_t axis = const_axes->at<loco::DataType::S32>(i);
      if (axis < 0)
        axis += node_tensor_shape.rank();
      axes_values.push_back(axis);
    }
  }
  else if (auto const_axes = dynamic_cast<loco::ConstGen *>(axes_node))
  {
    // TODO Support S64 type
    assert(const_axes->dtype() == loco::DataType::S32);

    for (uint32_t i = 0; i < const_axes->size<loco::DataType::S32>(); ++i)
    {
      int32_t axis = const_axes->at<loco::DataType::S32>(i);
      if (axis < 0)
        axis += node_tensor_shape.rank();
      axes_values.push_back(axis);
    }
  }
  else
    return false;

  // Create loco node to replace
  auto reduce = graph->nodes()->template create<loco::TensorReduce>();

  // replace
  reduce->func(reduceFunc<TFNode>());
  reduce->input(node->input());
  for (uint32_t i = 0; i < axes_values.size(); ++i)
    reduce->axes()->insert(axes_values.at(i));

  replace(node).with(reduce);

  INFO(l) << "TFNodeCanonicalize ReduceNode done";

  return true;
}

} // namespace

#endif // __TF_REDUCE_CANONICALIZE_HELPER_H__
