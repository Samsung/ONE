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

#include "moco/Pass/Passes/ConstantFoldStridedSlice.h"

#include "ConstantFoldHelper.h"
#include "TensorSliceEnumerator.h"

#include <moco/IR/Nodes/TFStridedSlice.h>
#include <moco/IR/Nodes/TFConst.h>

#include <moco/Support/NodeAs.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include <oops/UserExn.h>

#include <cassert>
#include <vector>

namespace
{

loco::TensorShape calc_output_shape(moco::TFStridedSlice *node)
{
  auto const_input = loco::must_cast<moco::TFConst *>(node->input());
  auto const_begin = loco::must_cast<moco::TFConst *>(node->begin());
  auto const_end = loco::must_cast<moco::TFConst *>(node->end());
  auto input_rank = const_input->rank();
  auto output_rank = input_rank;
  loco::TensorShape output_shape_range;

  output_shape_range.rank(input_rank);
  for (uint32_t r = 0; r < input_rank; ++r)
  {
    // TODO apply begin/end mask
    // TODO apply ellipsis mask
    // TODO apply strides
    auto end = const_end->at<loco::DataType::S32>(r);
    auto begin = const_begin->at<loco::DataType::S32>(r);
    auto size = end - begin;
    output_shape_range.dim(r).set(size);
  }

  loco::TensorShape output_tensor_shape;
  if (node->shrink_axis_mask() != 0)
  {
    for (uint32_t rs = 0; rs < input_rank; ++rs)
    {
      int32_t bit = 1 << rs;
      int32_t mask = node->shrink_axis_mask();
      if (bit & mask)
      {
        // shrink one dimension
        assert(output_rank > 0);
        output_rank = output_rank - 1;
      }
    }
    output_tensor_shape.rank(output_rank);
    for (uint32_t rs = 0, rd = 0; rs < input_rank; ++rs)
    {
      int32_t bit = 1 << rs;
      int32_t mask = node->shrink_axis_mask();
      if ((bit & mask) == 0)
      {
        // use this dimension
        output_tensor_shape.dim(rd).set(output_shape_range.dim(rs).value());
        rd++;
      }
      // else this dimension is shrink-ed
    }
  }
  else
  {
    output_tensor_shape = output_shape_range;
  }

  return output_tensor_shape;
}

moco::u32v_t vector_from_const(moco::TFConst *tfconst)
{
  moco::u32v_t result;

  auto rank = tfconst->rank();
  assert(rank == 1);
  auto dim = tfconst->dim(0).value();

  result.resize(dim);
  for (uint32_t r = 0; r < dim; ++r)
  {
    auto val = tfconst->at<loco::DataType::S32>(r);
    result.at(r) = val;
  }

  return result;
}

moco::u32v_t operator-(const moco::u32v_t &lhs, const moco::u32v_t &rhs)
{
  assert(lhs.size() == rhs.size());

  moco::u32v_t res;
  res.resize(lhs.size());
  for (uint32_t r = 0; r < lhs.size(); r++)
  {
    res.at(r) = lhs.at(r) - rhs.at(r);
  }
  return res;
}

template <typename T> T tfconst_at(const moco::TFConst *tfconst, const moco::u32v_t &pos);

template <> int32_t tfconst_at<int32_t>(const moco::TFConst *tfconst, const moco::u32v_t &pos)
{
  uint32_t rank = tfconst->rank();
  assert(rank == pos.size());
  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    uint32_t dim = tfconst->dim(r).value();
    element = element * dim + pos.at(r);
  }
  return tfconst->at<loco::DataType::S32>(element);
}

template <> float tfconst_at<float>(const moco::TFConst *tfconst, const moco::u32v_t &pos)
{
  uint32_t rank = tfconst->rank();
  assert(rank == pos.size());
  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    uint32_t dim = tfconst->dim(r).value();
    element = element * dim + pos.at(r);
  }
  return tfconst->at<loco::DataType::FLOAT32>(element);
}

void tfconst_at(moco::TFConst *tfconst, const moco::u32v_t &pos, int32_t value)
{
  // tfconst->rank() can be smaller than pos.size()
  // i.e., tfconst: shape[3] and pos[0,1]
  //                where shape[3] is output result shape
  //                [0,1] is position of input const
  uint32_t rank = pos.size();
  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    // this is like expand the shape from [3] to [1,3] to use same formula as in reading
    uint32_t dim = tfconst->rank() < r ? tfconst->dim(r).value() : 1;
    element = element * dim + pos.at(r);
  }

  tfconst->at<loco::DataType::S32>(element) = value;
}

void tfconst_at(moco::TFConst *tfconst, const moco::u32v_t &pos, float value)
{
  uint32_t rank = pos.size();
  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    uint32_t dim = tfconst->rank() < r ? tfconst->dim(r).value() : 1;
    element = element * dim + pos.at(r);
  }

  tfconst->at<loco::DataType::FLOAT32>(element) = value;
}

bool constantfold_stridedslice(moco::TFStridedSlice *node)
{
  auto const_input = dynamic_cast<moco::TFConst *>(node->input());
  if (const_input == nullptr)
  {
    // input is not TFConst, there's nothing to do
    return false;
  }

  // TODO support full mask features: see import codes also
  assert(node->begin_mask() == 0);
  assert(node->end_mask() == 0);
  assert(node->ellipsis_mask() == 0);
  assert(node->shrink_axis_mask() == 1);

  // TODO support other dtypes
  assert(const_input->dtype() == loco::DataType::S32 ||
         const_input->dtype() == loco::DataType::FLOAT32);

  auto const_begin = dynamic_cast<moco::TFConst *>(node->begin());
  auto const_end = dynamic_cast<moco::TFConst *>(node->end());
  auto const_strides = dynamic_cast<moco::TFConst *>(node->strides());
  if (const_begin == nullptr || const_end == nullptr || const_strides == nullptr)
  {
    return false;
  }

  // NOTE need shape but cannot depend on shape inference service module
  auto tensor_shape = calc_output_shape(node);
  auto input_shape = moco::tensor_shape(const_input);

  auto graph = node->graph();

  // Create our target TFConst node with shape from begin~end/strides
  auto const_sliced = moco::new_const(graph, tensor_shape, const_input->dtype());

  // Copy sliced elements using TensorSliceEnumerator
  moco::TensorSliceEnumerator etor;
  auto v_begin = vector_from_const(const_begin);
  auto v_end = vector_from_const(const_end);
  moco::u32v_t v_cursor;
  moco::u32v_t v_offset;

  etor.shape(input_shape);
  etor.begin(v_begin);
  etor.end(v_end);

  for (etor.start(); etor.valid(); etor.advance())
  {
    v_cursor = etor.cursor();
    v_offset = v_cursor - v_begin;

    if (const_input->dtype() == loco::DataType::S32)
    {
      int32_t value = tfconst_at<int32_t>(const_input, v_cursor);
      tfconst_at(const_sliced, v_offset, value);
    }
    else if (const_input->dtype() == loco::DataType::FLOAT32)
    {
      float value = tfconst_at<float>(const_input, v_cursor);
      tfconst_at(const_sliced, v_offset, value);
    }
  }

  // replace
  loco::replace(node).with(const_sliced);

  return true;
}

} // namespace

namespace moco
{

/**
 * @note This will Replace TFStridedSlice with TFConst when 'input' is TFConst
 *
 *       Before
 *                 A --- TFStridedSlice --- C
 *                 B --/
 *       After
 *                 A --- TFStridedSlice
 *                 B --/
 *                       TFConst ---------- C
 *       Where
 *                 A,B : inputs of TFStridedSlice
 *                 C : a node that uses TFStridedSlice as an input
 *                 TFStridedSlice is disconnected from C
 *                 Nodes are drawn multiple times to simplify the diagram
 *       Limits
 *                 Only limit set of inputs are supported for now
 */
bool ConstantFoldStridedSlice::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (auto sslice_node = as<moco::TFStridedSlice>(node))
    {
      if (constantfold_stridedslice(sslice_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace moco
