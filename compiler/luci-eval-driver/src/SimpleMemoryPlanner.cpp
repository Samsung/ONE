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

#include "SimpleMemoryPlanner.h"
#include <loco/IR/Algorithm.h>

namespace luci
{
namespace
{

constexpr uint32_t nodeNotAssigned = std::numeric_limits<int32_t>::max();

uint32_t computeOutputSize(Padding padding, uint32_t image_size, uint32_t filter_size,
                           uint32_t stride, uint32_t dilation_rate = 1)
{
  const int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding)
  {
    case Padding::SAME:
      return (image_size + stride - 1) / stride;
    case Padding::VALID:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      assert(false);
  }
}

}
uint32_t SimpleMemoryPlanner::PlanAllocations()
{
  PlanOrder();
  _required_size =  PlanOffset();
  DumpInform();
  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    luci::CircleNodeMemoryPlan memory_plan(i, _offsets[i]);
    luci::add_memory_plan(loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]), memory_plan);
  }
  return _required_size;
}

void SimpleMemoryPlanner::PlanOrder()
{
  // Get execution order in _ordered_nodes
  _ordered_nodes = loco::postorder_traversal(
    loco::output_nodes(const_cast<loco::Graph *>(_graph)));

  // Initialize vectors of first and last node for usage interval
  _alloc_node.assign(_ordered_nodes.size(), nodeNotAssigned);
  _dealloc_node.assign(_ordered_nodes.size(), nodeNotAssigned);

  // Vector for count usages
  std::vector<int> usages_counts(_ordered_nodes.size(), 0);

  auto allocate = [this](uint32_t node, uint32_t tensor) {
    if (_alloc_node[tensor] != nodeNotAssigned) {
      return;
    }
    assert(_dealloc_node[tensor] == nodeNotAssigned);
    _alloc_node[tensor] = node;
  };

  auto deallocate = [this](uint32_t node, uint32_t tensor) {
    assert(_dealloc_node[tensor] == nodeNotAssigned);
    _dealloc_node[tensor] = node;
  };

  // Increase refcounts for graph outputs and inputs nodes
  for (auto &output_node : output_nodes(_graph))
  {
    auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), output_node);
    size_t index = std::distance(_ordered_nodes.begin(), it);
    usages_counts[index]++;
  }

  for (auto &input_node : input_nodes(_graph))
  {
    auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), input_node);
    size_t index = std::distance(_ordered_nodes.begin(), it);
    usages_counts[index]++;
    allocate(0, index);
  }

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
    if (const auto *const_node = dynamic_cast<const luci::CircleConst *>(node))
    {
      allocate(0, i);
    }
    allocate(i, i);

    auto prev_nodes = preds(node);
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

void SimpleMemoryPlanner::GetSizesInformation(uint32_t required_size)
{
  size_t consts_size = 0;
  size_t input_size = 0;

  for (const auto &alloc : _alloc_inform_vector)
  {
    if (const auto *const_node =
          dynamic_cast<const luci::CircleConst *>(_ordered_nodes[alloc.node_num]))
    {
      consts_size += alloc.size;
      required_size -= alloc.size;
    }
  }

  for (auto &input_node : input_nodes(_graph))
  {
    auto it = std::find(_ordered_nodes.begin(), _ordered_nodes.end(), input_node);
    size_t input_index = std::distance(_ordered_nodes.begin(), it);

    for (const auto &alloc : _alloc_inform_vector)
    {
      if (alloc.node_num == input_index)
      {
        input_size += alloc.size;
      }
    }
  }

  printf("Size of inputs is = %ld\n", input_size);
  printf("Size of constant nodes = %ld\n", consts_size);
  printf("Required size of greedy_by_size approach without constant nodes is = %d\n", required_size);
  printf("Required size of greedy_by_size approach without constant and input nodes is = %ld\n", required_size - input_size);
}

// Get required sizes for naive approach and greedy by size offset calculation approach
// Get offsets for greedy by size approach and fill offsets vector
uint32_t SimpleMemoryPlanner::PlanOffset()
{
  // To have required size for naive approach
  // auto naive_size = NaiveSize();
  auto greedy_by_size = GreedyBySize();

  _offsets.resize(_ordered_nodes.size());
  for (const auto & alloc : _alloc_inform_vector)
  {
    // Fill offsets vector: the last one should be offsets for temporaries tensors
    if (alloc.is_temp)
    {
      _offsets[alloc.node_num].push_back(alloc.offset);
    } else {
      _offsets[alloc.node_num].insert(_offsets[alloc.node_num].begin(), alloc.offset);
    }
  }

  printf("Required size with greedy_by_size approach = %d\n", greedy_by_size);

  GetSizesInformation(greedy_by_size);

  return greedy_by_size;
}

// null_consts = true - size of consts nodes will be equal 0;
// null_inputs = true - size of inputs nodes will be equal 0;
// null_im2col = true - size of im2col nodes will be equal 0;
void SimpleMemoryPlanner::CreateAllocVector(bool null_consts,
                                            bool null_inputs,
                                            bool null_im2col) {
  auto node_compare = [this](const AllocNodeInform &alloc_1, const AllocNodeInform &alloc_2) {
    auto idx1 = alloc_1.node_num;
    auto idx2 = alloc_2.node_num;

    if (this->_alloc_node[idx1] == 0 &&
        this->_dealloc_node[idx1] == nodeNotAssigned) {
      if (this->_alloc_node[idx2] == 0 &&
          this->_dealloc_node[idx2] == nodeNotAssigned) {
        return idx1 < idx2;
      }
      return true;
    }
    if (this->_alloc_node[idx2] == 0 &&
        this->_dealloc_node[idx2] == nodeNotAssigned) {
      return false;
    }

    auto size_1 = alloc_1.size;
    auto size_2 = alloc_2.size;

    if (size_1 != size_2) {
      return size_1 > size_2;
    }
    return this->_alloc_node[idx1] < this->_alloc_node[idx2];
  };

  _alloc_inform_vector.resize(_ordered_nodes.size());

  for (size_t i = 0; i < _ordered_nodes.size(); i++)
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(_ordered_nodes[i]);
    auto node_size = 1;
    for (uint32_t axis = 0; axis < circle_node->rank(); ++axis)
    {
      node_size *= circle_node->dim(axis).value();
    }
    node_size *= size(circle_node->dtype());

    _alloc_inform_vector[i].node_num = i;
    _alloc_inform_vector[i].first_node = _alloc_node[i];
    _alloc_inform_vector[i].last_node = _dealloc_node[i];

    const auto *const_node = dynamic_cast<const luci::CircleConst *>(circle_node);
    if (i == 0 && null_inputs)
    {
      _alloc_inform_vector[i].size = 0;
    } else if (const_node && null_consts) {
      _alloc_inform_vector[i].size = 0;
    } else {
      _alloc_inform_vector[i].size = node_size;
    }

    // Im2col
    auto opcode = circle_node->opcode();
    if (opcode == luci::CircleOpcode::CONV_2D)
    {
      auto conv = loco::must_cast<const luci::CircleConv2D *>(circle_node);
      auto im2col_size = Im2colSize(conv);
      if (im2col_size > 0)
      {
        AllocNodeInform temp_alloc;

        if (null_im2col)
        {
          temp_alloc.size = 0;
        } else {
          temp_alloc.size = im2col_size;
        }

        temp_alloc.first_node = i - 1;
        temp_alloc.last_node = i + 1;
        temp_alloc.node_num = i;
        temp_alloc.is_temp = true;

        _alloc_inform_vector.push_back(temp_alloc);
        _alloc_node.push_back(i);
        _dealloc_node.push_back(i);
      }
    }
  }
  // Sort alloc_inform_vector with node_compare for the greedy by size approach.
  std::sort(_alloc_inform_vector.begin(), _alloc_inform_vector.end(), node_compare);
}

uint32_t  SimpleMemoryPlanner::GreedyBySize()
{
  size_t result_size = 0;
  CreateAllocVector(false, false, false);
  std::vector<AllocNodeInform> ordered_alloc_inform;
  for (auto& current_node : _alloc_inform_vector)
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

    auto insertion_it = std::upper_bound(ordered_alloc_inform.begin(),
                                         ordered_alloc_inform.end(), current_node);
    ordered_alloc_inform.insert(insertion_it, current_node);
  }
  return result_size;
}

uint32_t SimpleMemoryPlanner::NaiveSize()
{
  uint32_t i = 0;
  int32_t required_size = 0;
  int32_t offset = 0;

  for (auto &node : _ordered_nodes)
  {
    uint32_t current_size = 1;
    uint32_t im2col_size = 0;
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    for (uint32_t axis = 0; axis < circle_node->rank(); ++axis)
    {
      current_size *= circle_node->dim(axis).value();
    }
    current_size *= size(circle_node->dtype());
    auto opcode = circle_node->opcode();
    if (opcode == luci::CircleOpcode::CONV_2D)
    {
      auto conv = loco::must_cast<const luci::CircleConv2D *>(node);

      im2col_size += Im2colSize(conv);
    }

    required_size += current_size + im2col_size;

    offset += current_size + im2col_size;
    i++;
  }
  return required_size;
}

uint32_t SimpleMemoryPlanner::Im2colSize(const luci::CircleConv2D *conv)
{
  auto conv_input = loco::must_cast<luci::CircleNode *>(conv->input());
  auto filter = loco::must_cast<luci::CircleNode *>(conv->filter());
  auto padding = (conv->padding());
  uint32_t stride_height = conv->stride()->h();
  uint32_t stride_width = conv->stride()->w();

  uint32_t dilation_height_factor = conv->dilation()->h();
  uint32_t dilation_width_factor = conv->dilation()->w();

  uint32_t filter_height = filter->dim(1).value();
  uint32_t filter_width = filter->dim(2).value();

  const bool need_dilated_im2col =
    dilation_height_factor != 1 || dilation_width_factor != 1;
  const bool need_non_dilated_im2col = stride_height != 1 || stride_width != 1 ||
                                       filter_height != 1 || filter_width != 1;
  bool need_im2col =
    conv_input->dtype() != loco::DataType::S16 && (need_dilated_im2col || need_non_dilated_im2col);

  if (!need_im2col) { return 0; }

  uint32_t input_depth = conv_input->dim(3).value();
  uint32_t input_height = conv_input->dim(1).value();
  uint32_t input_width = conv_input->dim(2).value();

  uint32_t output_height =
    computeOutputSize(padding, input_height, filter_height, stride_height,
                      dilation_height_factor);
  uint32_t output_width =
    computeOutputSize(padding, input_width, filter_width, stride_width,
                      dilation_width_factor);

  uint32_t batches = conv_input->dim(0).value();

  return batches * output_height * output_width * input_depth * filter_height * filter_width * size(conv_input->dtype());
}

void SimpleMemoryPlanner::DumpInform()
{
  uint32_t max_breadth = 0;

  for (uint32_t i = 0; i < _ordered_nodes.size(); i++)
  {
    auto current_node_it = std::find_if(_alloc_inform_vector.begin(), _alloc_inform_vector.end(), [this, i](const AllocNodeInform &x){
      return x.node_num == i && !x.is_temp; });
    for (uint32_t j = 0; j < _ordered_nodes.size(); j++)
    {
      auto first_node = _alloc_node[j];
      auto last_node = _dealloc_node[j];

      auto it = std::find_if(_alloc_inform_vector.begin(), _alloc_inform_vector.end(), [this, j](const AllocNodeInform &x){
        return x.node_num == j && !x.is_temp; });
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
//    printf("i = %d   name = %s    with size = %d    with offset = %d    with breadth = %u\n "
//          "first_node = %d   last_node = %d\n",
//           i, node->name().c_str(), current_node_it->size,
//           current_node_it->offset, current_node_it->breadth,
//           current_node_it->first_node, current_node_it->last_node);
  }
  printf("Lower bound is = %u\n", max_breadth);
  std::sort(_alloc_inform_vector.begin(), _alloc_inform_vector.end(), [](const AllocNodeInform &first,
                                                                         const AllocNodeInform &second){
    if (first.breadth != second.breadth)
      return first.breadth > second.breadth;
    return first.node_num < second.node_num;
  });

  for (const auto alloc_inform : _alloc_inform_vector)
  {
    if (alloc_inform.is_temp)
    {
      continue;
    }
    auto node = loco::must_cast<luci::CircleNode *>(_ordered_nodes.at(alloc_inform.node_num));
    printf("i = %d   name = %s    with breadth = %u\n",//name = %s    with size = %d    with offset = %d    with breadth = %u\n ",
           alloc_inform.node_num, node->name().c_str(), alloc_inform.breadth); //node->name().c_str(), alloc_inform.size, alloc_inform.offset, alloc_inform.breadth);
  }


}

} // namespace luci_interpreter