/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_ACL_COMMON_ACL_SUB_TENSOR_ANALYZER_H__
#define __ONERT_BACKEND_ACL_COMMON_ACL_SUB_TENSOR_ANALYZER_H__

#include <cl_common/ParentInfo.h>

#include <ir/OperationVisitor.h>
#include <ir/Graph.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

/**
 * @brief Class to analyze tensor subsumption
 */
class AclSubTensorAnalyzer : public ir::OperationVisitor
{
public:
  /**
   * @brief     Construct a new SubTensorAnalyzer object
   * @param[in] ctx Graph operand set
   */
  AclSubTensorAnalyzer(const ir::Graph &graph) : _graph{graph}
  {
    // DO NOTHING
  }

public:
  void setUsePadding() { usePadding = true; }

  void visit(const ir::operation::Concat &node) override
  {
    //  If operator is concat, fill subsumption info
    int32_t axis_raw = node.param().axis;

    const auto &output_index = node.getOutputs().at(0);
    const auto &inputs = node.getInputs();

    int32_t axis_point = 0;
    const auto rank = _graph.operands().at(output_index).shape().rank();
    int32_t axis = axis_raw < 0 ? (axis_raw + rank) : axis_raw;
    assert(rank > axis);

    // Concat elimination when axis is last dimension is not supported
    // https://github.com/Samsung/ONE/issues/4407
    // TODO Enable if backend don't use padding
    if ((axis == rank - 1) && usePadding)
      return;

    for (const auto &ind : inputs)
    {
      /**
       * NOTE Not support below cases.
       * 1. concat's input is a constant.
       * 2. concat's input is a input of model.
       * 3. concat's input already becomes a subtensor of another concat.
       */
      if (_graph.operands().at(ind).isConstant() || _graph.getInputs().contains(ind) ||
          _parent_map.find(ind) != _parent_map.end())
      {
        return;
      }
    }

    for (const auto &input_index : inputs)
    {
      auto input_shape = _graph.operands().at(input_index).shape();
      auto input_layout = _graph.operands().at(input_index).info().layout();
      assert(rank == input_shape.rank());

      ir::Coordinates coordinate_info{};
      for (int i = 0; i < rank; i++)
      {
        coordinate_info.set(i, 0);
      }
      coordinate_info.set(axis, axis_point);

      _parent_map.emplace(input_index,
                          cl_common::ParentInfo{output_index, input_layout, coordinate_info});

      axis_point += input_shape.dim(axis);
    }
  }

  std::unordered_map<ir::OperandIndex, cl_common::ParentInfo> &&releaseParentMap()
  {
    return std::move(_parent_map);
  }

private:
  const ir::Graph &_graph;
  std::unordered_map<ir::OperandIndex, cl_common::ParentInfo> _parent_map;
  bool usePadding{false};
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_ACL_SUB_TENSOR_ANALYZER_H__
