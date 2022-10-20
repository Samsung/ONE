/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExecutionPlanner.h"
#include <loco/IR/Algorithm.h>
#include <luci/UserSettings.h>

#include <json.h>
#include <fstream>

#include <limits> // std::numeric_limits

namespace circle_planner
{
namespace
{

constexpr uint32_t node_not_assigned = std::numeric_limits<int32_t>::max();

bool isExecutableNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // The following nodes denote outputs of multiple-output nodes.
    // The list is synchronized with the same list from luci-interpreter/src/loader/GraphLoader.cpp
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      return false;
    default:
      return true;
  }
}

bool isTensorProducingNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // The following nodes are multiple-output nodes. They do not produce tensors, the tensors
    // are produced by the corresponding *Out nodes instead.
    // The list is synchronized with the same list from luci-interpreter/src/loader/GraphLoader.cpp
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::UNPACK:
      return false;
    default:
      return true;
  }
}

// Create allocation node part for current circle node for json allocation info file
void create_allocation_node(Json::Value &allocations_node,
                            AllocationNodeInformation &alloca_node_inform, uint32_t alive_till_max,
                            luci::CircleNode *circle_node)
{
  Json::Value allocation_node;
  if (alloca_node_inform.size == 0)
    return;

  allocation_node["offset"] = alloca_node_inform.offset;
  allocation_node["size"] = alloca_node_inform.size;
  allocation_node["alive_from"] = alloca_node_inform.first_node;

  if (alloca_node_inform.last_node == node_not_assigned)
    allocation_node["alive_till"] = alive_till_max + 1;
  else
    allocation_node["alive_till"] = alloca_node_inform.last_node;

  allocation_node["origin"] = circle_node->name();

  allocations_node.append(allocation_node);
}

// TODO: Introduce inplace optimization
bool can_be_inplace_optimization_node(luci::CircleNode *node)
{
  switch (node->opcode())
  {
    case luci::CircleOpcode::LOGISTIC:
    case luci::CircleOpcode::RESHAPE:
    case luci::CircleOpcode::EXPAND_DIMS:
      return true;
    default:
      return false;
  }
}

} // namespace

void ExecutionPlanner::make_execution_plan_luci_interpreter()
{
  get_default_execution_order_plan();
  _required_size = get_offsets_with_greedy_by_size();

  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    luci::CircleNodeExecutionPlan execution_plan(i, _offsets[i]);
    luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                             execution_plan);
  }

  printf("Buffer required memory = %d\n", _required_size);
  dump_inform();
}

void ExecutionPlanner::make_execution_plan_onert_micro_base()
{
  switch (_buffers_type)
  {
    case SupportedBuffersType::COMMON:
      make_execution_plan_onert_micro_common_buffer();
      break;
    case SupportedBuffersType::SPLIT:
      make_execution_plan_onert_micro_split_buffer();
      break;
    default:
      throw std::runtime_error("Unsupported buffer type\n");
  }
}

void ExecutionPlanner::make_execution_plan_onert_micro_split_buffer()
{
  const auto input_size = _graph->inputs()->size();
  const auto output_size = _graph->outputs()->size();

  // Make execution plan for inputs
  _ordered_nodes = loco::input_nodes(_graph);
  const auto input_required_size = get_offsets_with_greedy_by_size();

  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    luci::CircleNodeExecutionPlan execution_plan(i, _offsets[i]);
    luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                             execution_plan);
  }
  dump_inform();
  printf("Input graph buffer required memory = %d\n", input_required_size);

  // Clear structures for next buffer
  _ordered_nodes.clear();
  _alloc_node_inform_vector.clear();
  _dealloc_node.clear();
  _alloc_node.clear();
  _offsets.clear();
  _required_size = 0;

  // Make execution plan for outputs
  _ordered_nodes = loco::output_nodes(_graph);
  const auto output_required_size = get_offsets_with_greedy_by_size();
  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    luci::CircleNodeExecutionPlan execution_plan(i + input_size, _offsets[i]);
    luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                             execution_plan);
  }
  dump_inform();
  printf("Output graph buffer required memory = %d\n", output_required_size);

  // Clear structures for next buffer
  _ordered_nodes.clear();
  _alloc_node_inform_vector.clear();
  _dealloc_node.clear();
  _alloc_node.clear();
  _offsets.clear();
  _required_size = 0;

  // Make execution plan for intermediates calculations
  get_default_execution_order_plan_without_inputs_and_outputs();
  const auto main_graph_required_size = get_offsets_with_greedy_by_size();

  int counter_ops = 0;
  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    const auto circle_node = dynamic_cast<luci::CircleNode *>(_ordered_nodes[i]);
    if (circle_node->opcode() != luci::CircleOpcode::CIRCLECONST and
        circle_node->opcode() != luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
    {
      luci::CircleNodeExecutionPlan execution_plan(counter_ops + input_size + output_size, _offsets[i]);
      luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                               execution_plan);
      counter_ops++;
    }
  }
  dump_inform();
  printf("Main graph buffer required memory = %d\n", main_graph_required_size);
}

void ExecutionPlanner::make_execution_plan_onert_micro_common_buffer()
{
  get_default_execution_order_plan();
  _required_size = get_offsets_with_greedy_by_size();

  // Find prev nodes for output nodes (actual graph output node, not luci::CircleOutput)
  const auto output_nodes = loco::output_nodes(const_cast<loco::Graph *>(_graph));
  std::vector<loco::Node *> output_prev_nodes;
  for (const auto output_node : output_nodes)
  {
    const auto prev_nodes = loco::preds(output_node);
    std::copy(prev_nodes.begin(), prev_nodes.end(), std::back_inserter(output_prev_nodes));
  }
  const auto output_nodes_size = output_prev_nodes.size();

  const auto inputs_nodes = loco::input_nodes(_graph);
  const auto input_nodes_size = inputs_nodes.size();

  int32_t counter_ops = 0;
  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    const auto circle_node = dynamic_cast<luci::CircleNode *>(_ordered_nodes[i]);
    // First write to input nodes
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
    {
      // Find input_position for proper position in execution order
      const auto input_position = std::distance(inputs_nodes.begin(), std::find(inputs_nodes.begin(), inputs_nodes.end(), circle_node));
      luci::CircleNodeExecutionPlan execution_plan(input_position, _offsets[i]);
      luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                               execution_plan);
    }
    // Second write to actual output nodes (not luci::CircleOutput)
    else if (std::find(output_prev_nodes.begin(), output_prev_nodes.end(), circle_node) != output_prev_nodes.end())
    {
      // Find output_position for proper position in execution order
      const auto output_position = std::distance(output_prev_nodes.begin(), std::find(output_prev_nodes.begin(), output_prev_nodes.end(), circle_node));
      luci::CircleNodeExecutionPlan execution_plan(input_nodes_size + output_position, _offsets[i]);
      luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                               execution_plan);
    }
    // Finally write to all intermediate nodes
    else if (circle_node->opcode() != luci::CircleOpcode::CIRCLECONST and
             circle_node->opcode() != luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
    {
      luci::CircleNodeExecutionPlan execution_plan(counter_ops + input_nodes_size + output_nodes_size, _offsets[i]);
      luci::add_execution_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]),
                               execution_plan);
      counter_ops++;
    }
  }

  dump_inform();
  printf("Buffer required memory = %d\n", _required_size);
}

void ExecutionPlanner::make_execution_plan()
{
  switch (_runtime_type)
  {
    case ONERT_MICRO:
      make_execution_plan_onert_micro_base();
      break;
    case LUCI_INTERPRETER:
      make_execution_plan_luci_interpreter();
      break;
    default:
      throw std::runtime_error("Unsupported runtime platform\n");
  }

  auto settings = luci::UserSettings::settings();
  settings->set(luci::UserSettings::Key::ExecutionPlanGen, true);
}

void ExecutionPlanner::create_json_allocation_file(const std::string &json_path)
{
  Json::Value main_tree;
  Json::Value segments_node;
  Json::Value allocations_node;

  uint32_t alive_till_max = 0;

  // Find max dealloc value to assign to nodes with node_not_assigned value
  for (const auto elem : _dealloc_node)
  {
    if (alive_till_max < elem and elem != node_not_assigned)
      alive_till_max = elem;
  }

  for (auto &alloc_node_inform : _alloc_node_inform_vector)
  {
    const auto node_num = alloc_node_inform.node_num;
    const auto circle_node = loco::must_cast<luci::CircleNode *>(_ordered_nodes[node_num]);

    create_allocation_node(allocations_node, alloc_node_inform, alive_till_max, circle_node);
  }

  // Create segment part
  Json::Value segment_node;
  segment_node["name"] = "Segment1";
  segment_node["allocations"] = allocations_node;
  segments_node.append(segment_node);

  main_tree["schema_version"] = 1;
  main_tree["segments"] = segments_node;

  Json::StreamWriterBuilder builder;
  const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

  // Write to json file
  std::ofstream out;
  out.open(json_path);
  if (out.is_open())
  {
    writer->write(main_tree, &out);
  }
}

void ExecutionPlanner::get_default_execution_order_plan()
{
  // Get execution order in _ordered_nodes
  _ordered_nodes = loco::postorder_traversal(loco::output_nodes(const_cast<loco::Graph *>(_graph)));
}

void ExecutionPlanner::get_default_execution_order_plan_without_inputs_and_outputs()
{
  // Get all nodes
  _ordered_nodes = loco::postorder_traversal(loco::output_nodes(const_cast<loco::Graph *>(_graph)));

  // Get real output nodes (not luci::CircleOutput)
  const auto output_nodes = loco::output_nodes(const_cast<loco::Graph *>(_graph));
  std::vector<loco::Node *> output_prev_nodes;
  for (const auto output_node : output_nodes)
  {
    const auto prev_nodes = loco::preds(output_node);
    std::copy(prev_nodes.begin(), prev_nodes.end(), std::back_inserter(output_prev_nodes));
  }

  // Remove input and real output nodes from _ordered_nodes
  _ordered_nodes.erase(std::remove_if(_ordered_nodes.begin(), _ordered_nodes.end(),
                                      [&output_prev_nodes](auto node)
                                      {
                                        const auto circle_node = dynamic_cast<luci::CircleNode *>(node);

                                        return circle_node->opcode() == luci::CircleOpcode::CIRCLEINPUT or
                                               circle_node->opcode() == luci::CircleOpcode::CIRCLEOUTPUT or
                                               std::find(output_prev_nodes.begin(), output_prev_nodes.end(), node) != output_prev_nodes.end();

                                      }), _ordered_nodes.end());
}

void ExecutionPlanner::get_usage_interval()
{
  // Initialize vectors of first and last nodes for usage interval
  _alloc_node.assign(_ordered_nodes.size(), node_not_assigned);
  _dealloc_node.assign(_ordered_nodes.size(), node_not_assigned);

  // Vector for count usages
  std::vector<int> usages_counts(_ordered_nodes.size(), 0);

  auto allocate = [this](uint32_t node, uint32_t tensor) {
    if (_alloc_node[tensor] != node_not_assigned)
    {
      return;
    }
    assert(_dealloc_node[tensor] == node_not_assigned);
    _alloc_node[tensor] = node;
  };

  auto deallocate = [this](uint32_t node, uint32_t tensor) {
    assert(_dealloc_node[tensor] == node_not_assigned);
    _dealloc_node[tensor] = node;
  };

  // Increase refcounts for graph outputs and inputs nodes
  for (auto &output_node : output_nodes(_graph))
  {
    auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), output_node);
    if (it == _ordered_nodes.end())
      continue;
    size_t index = std::distance(_ordered_nodes.begin(), it);
    usages_counts[index]++;
  }

  for (auto &input_node : input_nodes(_graph))
  {
    auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), input_node);
    if (it == _ordered_nodes.end())
      continue;
    size_t index = std::distance(_ordered_nodes.begin(), it);
    usages_counts[index]++;
    allocate(0, index);
  }

  // Increase refcounts of usage for all nodes in _ordered_nodes vector
  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    const auto node = _ordered_nodes.at(i);
    auto prev_nodes = preds(node);
    for (auto &prev_node : prev_nodes)
    {
      auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), prev_node);
      size_t index = std::distance(_ordered_nodes.begin(), it);
      usages_counts[index]++;
    }
  }

  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    const auto node = _ordered_nodes.at(i);
    auto prev_nodes = preds(node);
    if (const auto *const_node = dynamic_cast<const luci::CircleConst *>(node))
    {
      allocate(0, i);
    }
    else if (!isExecutableNode(loco::must_cast<luci::CircleNode *>(node)))
    {
      // If current node is multi output node than begin life time for current node should start
      // when prev node start live
      auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), *prev_nodes.begin());
      size_t index = std::distance(_ordered_nodes.begin(), it);
      allocate(index, i);
    }
    else
    {
      allocate(i, i);
    }

    for (auto &prev_node : prev_nodes)
    {
      auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), prev_node);
      size_t index = std::distance(_ordered_nodes.begin(), it);
      usages_counts[index]--;
      if (usages_counts[index] == 0)
      {
        deallocate(i, index);
      }
    }
  }
}

uint32_t ExecutionPlanner::get_offsets_with_greedy_by_size()
{
  get_usage_interval();
  auto required_size = greedy_by_size_approach();

  _offsets.resize(_ordered_nodes.size());
  for (const auto &alloc : _alloc_node_inform_vector)
  {
    // Fill offsets vector: first go offset for current node and then should go offsets for
    // temporaries tensors
    if (alloc.is_temp)
    {
      _offsets[alloc.node_num].push_back(alloc.offset);
    }
    else
    {
      _offsets[alloc.node_num].insert(_offsets[alloc.node_num].begin(), alloc.offset);
    }
  }
  return required_size;
}

uint32_t ExecutionPlanner::greedy_by_size_approach()
{
  size_t result_size = 0;
  create_alloc_node_inform_vector(_is_null_consts, _is_null_inputs, _is_null_scratchpads);
  std::vector<AllocationNodeInformation> ordered_alloc_inform;
  for (auto &current_node : _alloc_node_inform_vector)
  {
    if (current_node.size == 0)
    {
      current_node.offset = 0;
      continue;
    }
    const uint32_t offsetNotAssigned = std::numeric_limits<uint32_t>::max();
    size_t best_offset = offsetNotAssigned;
    uint32_t best_offset_fit = offsetNotAssigned;

    uint32_t current_offset = 0;

    for (const auto &alloc_inform : ordered_alloc_inform)
    {
      if ((alloc_inform.last_node < current_node.first_node ||
           alloc_inform.first_node > current_node.last_node))
      {
        continue;
      }

      if (current_offset + current_node.size <= alloc_inform.offset &&
          alloc_inform.offset - current_offset < best_offset_fit)
      {
        best_offset = current_offset;
        best_offset_fit = alloc_inform.offset - current_offset;
      }
      current_offset = std::max(current_offset, alloc_inform.offset + alloc_inform.size);
    }
    if (best_offset == offsetNotAssigned)
    {
      best_offset = current_offset;
    }

    result_size = std::max(result_size, best_offset + current_node.size);
    current_node.offset = best_offset;

    auto insertion_it =
      std::upper_bound(ordered_alloc_inform.begin(), ordered_alloc_inform.end(), current_node);
    ordered_alloc_inform.insert(insertion_it, current_node);
  }
  return result_size;
}

void ExecutionPlanner::create_alloc_node_inform_vector(bool null_consts, bool null_inputs,
                                                       bool null_scratchpad)
{
  auto node_compare = [this](const AllocationNodeInformation &alloc_1,
                             const AllocationNodeInformation &alloc_2) {
    auto idx1 = alloc_1.node_num;
    auto idx2 = alloc_2.node_num;

    if (this->_alloc_node[idx1] == 0 && this->_dealloc_node[idx1] == node_not_assigned)
    {
      if (this->_alloc_node[idx2] == 0 && this->_dealloc_node[idx2] == node_not_assigned)
      {
        return idx1 < idx2;
      }
      return true;
    }
    if (this->_alloc_node[idx2] == 0 && this->_dealloc_node[idx2] == node_not_assigned)
    {
      return false;
    }

    auto size_1 = alloc_1.size;
    auto size_2 = alloc_2.size;

    if (size_1 != size_2)
    {
      return size_1 > size_2;
    }
    return this->_alloc_node[idx1] < this->_alloc_node[idx2];
  };

  _alloc_node_inform_vector.resize(_ordered_nodes.size());

  for (size_t i = 0; i < _ordered_nodes.size(); i++)
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]);
    auto node_size = circle_node->rank() > 0 ? 1 : 0;
    for (uint32_t axis = 0; axis < circle_node->rank(); ++axis)
    {
      node_size *= circle_node->dim(axis).value();
    }
    node_size *= size(circle_node->dtype());

    _alloc_node_inform_vector[i].node_num = i;
    _alloc_node_inform_vector[i].first_node = _alloc_node[i];
    _alloc_node_inform_vector[i].last_node = _dealloc_node[i];

    const auto *const_node = dynamic_cast<const luci::CircleConst *>(circle_node);
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEINPUT && null_inputs)
    {
      _alloc_node_inform_vector[i].size = 0;
    }
    else if (const_node && null_consts)
    {
      _alloc_node_inform_vector[i].size = 0;
    }
    else if (!isTensorProducingNode(circle_node))
    {
      _alloc_node_inform_vector[i].size = 0;
    }
    else
    {
      _alloc_node_inform_vector[i].size = node_size;
    }

    // Scratchpad If needed
    std::vector<uint32_t> scratchpad_sizes;
    if (!null_scratchpad)
    {
      switch (circle_node->opcode())
      {
        case luci::CircleOpcode::AVERAGE_POOL_2D:
        {
          const auto avg_pool = loco::must_cast<const luci::CircleAveragePool2D *>(circle_node);
          scratchpad_sizes.push_back(
            _scratchpad_helper->ComputeScratchpadSizeAveragePool2d(avg_pool));
          break;
        }
        case luci::CircleOpcode::BATCH_MATMUL:
        {
          const auto batch_mat_mul = loco::must_cast<const luci::CircleBatchMatMul *>(circle_node);
          scratchpad_sizes = _scratchpad_helper->ComputeScratchpadSizeBatchMatMul(batch_mat_mul);
          break;
        }
        case luci::CircleOpcode::CONV_2D:
        {
          const auto conv = loco::must_cast<const luci::CircleConv2D *>(circle_node);
          scratchpad_sizes.push_back(_scratchpad_helper->ComputeScratchpadSizeConv2d(conv));
          break;
        }
        case luci::CircleOpcode::DEPTHWISE_CONV_2D:
        {
          const auto depthwise_conv =
            loco::must_cast<const luci::CircleDepthwiseConv2D *>(circle_node);
          scratchpad_sizes.push_back(
            _scratchpad_helper->ComputeScratchpadSizeDepthwiseConv2d(depthwise_conv));
          break;
        }
        case luci::CircleOpcode::SVDF:
        {
          const auto svdf = loco::must_cast<const luci::CircleSVDF *>(circle_node);
          scratchpad_sizes = _scratchpad_helper->ComputeScratchpadSizeSVDF(svdf);
          break;
        }
        default:
          break;
      }
    }

    for (const auto scratchpad_size : scratchpad_sizes)
    {
      if (scratchpad_size > 0)
      {
        AllocationNodeInformation temp_alloc;

        temp_alloc.size = scratchpad_size;
        temp_alloc.first_node = i - 1;
        temp_alloc.last_node = i + 1;
        temp_alloc.node_num = i;
        temp_alloc.is_temp = true;

        _alloc_node_inform_vector.push_back(temp_alloc);
        _alloc_node.push_back(i);
        _dealloc_node.push_back(i);
      }
    }
  }
  // Sort _alloc_node_inform_vector with node_compare for the greedy by size approach.
  std::sort(_alloc_node_inform_vector.begin(), _alloc_node_inform_vector.end(), node_compare);
}

void ExecutionPlanner::dump_inform()
{
  uint32_t max_breadth = 0;

  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    auto current_node_it = std::find_if(
      _alloc_node_inform_vector.begin(), _alloc_node_inform_vector.end(),
      [i](const AllocationNodeInformation &x) { return x.node_num == i && !x.is_temp; });
    for (uint32_t j = 0; j < _ordered_nodes.size(); j++)
    {
      auto first_node = _alloc_node[j];
      auto last_node = _dealloc_node[j];

      auto it = std::find_if(
        _alloc_node_inform_vector.begin(), _alloc_node_inform_vector.end(),
        [j](const AllocationNodeInformation &x) { return x.node_num == j && !x.is_temp; });
      if (i >= first_node && i <= last_node)
      {
        current_node_it->breadth += it->size;
      }
    }
    if (max_breadth < current_node_it->breadth)
    {
      max_breadth = current_node_it->breadth;
    }

    auto node = loco::must_cast<luci::CircleNode *>(_ordered_nodes.at(i));
    printf("node_num = %d   node_name = %s    node_size = %d    node_offset = %d  node_breadth = "
           "%u node_first_node = %d   node_last_node = %d\n",
           i, node->name().c_str(), current_node_it->size, current_node_it->offset,
           current_node_it->breadth, current_node_it->first_node, current_node_it->last_node);
  }
  printf("Lower bound is = %u\n", max_breadth);
  std::sort(_alloc_node_inform_vector.begin(), _alloc_node_inform_vector.end(),
            [](const AllocationNodeInformation &first, const AllocationNodeInformation &second) {
              if (first.breadth != second.breadth)
                return first.breadth > second.breadth;
              return first.node_num < second.node_num;
            });
}

} // namespace circle_planner
