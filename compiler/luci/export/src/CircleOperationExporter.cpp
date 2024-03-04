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

#include "CircleOperationExporter.h"
#include "CircleOperationExporterRule.h"

#include <luci/IR/CircleNode.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Plan/CircleNodeExecutionPlan.h>
#include <luci/Plan/CircleMapTensorsIndexes.h>
#include <loco/IR/Algorithm.h>

namespace luci
{

void exportNodes(loco::Graph *g, flatbuffers::FlatBufferBuilder &builder, SerializedModelData &md,
                 SerializedGraphData &gd)
{
  uint32_t node_position = 0;
 // std::map<uint32_t, uint32_t> tensors_indexes;
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    ExportContext ctx{builder, md, gd};
    OperationExporterRule exporter_rule{ctx};

    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&exporter_rule);

    const auto ops_size = gd._operators.size();

    if (has_origin(circle_node) && ops_size != gd._operators.size())
    {
      const auto node_id = gd._operators.size() - 1;
      for (auto source : get_origin(circle_node)->sources())
      {
        md._metadata.add_op_table(node_id, source->id());
      }
    }
    if (has_execution_plan(circle_node))
    {
      // Add to node (in node_position) metadata vector with execution_plan information:
      // order of execution, and offsets output tensors.
      const auto execution_plan = get_execution_plan(circle_node);
      std::vector<uint32_t> execution_plan_vector;
      execution_plan_vector.push_back(execution_plan.order_in_plan());
      for (auto offset : execution_plan.offsets())
      {
        execution_plan_vector.push_back(offset);
      }
      md._metadata.add_execution_plan_table(node_position, execution_plan_vector);
    }

//    if (has_map_tensors_index(circle_node))
//    {
//      // Add to node (in node_position) metadata vector with execution_plan information:
//      // order of execution, and offsets output tensors.
//      const auto map_tensors = get_map_tensors_index(circle_node);
//      tensors_indexes[map_tensors.first_idx()] = map_tensors.second_idx();
//
//    }

    node_position++;
  }

//  if (!tensors_indexes.empty())
//  {
//    md._metadata.map_tensors_indexes(tensors_indexes);
//  }

}

} // namespace luci
