/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "luci/Service/CircleShapeInferenceRule.h"
#include "Check.h"

#include "CircleShapeInferenceHelper.h"
#include "ShapeInfer_StridedSlice.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <oops/InternalExn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace
{

std::ostream &operator<<(std::ostream &os, const loco::TensorShape &tensor_shape)
{
  os << "[";
  for (uint32_t r = 0; r < tensor_shape.rank(); ++r)
  {
    if (r)
      os << ",";

    if (tensor_shape.dim(r).known())
      os << tensor_shape.dim(r).value();
    else
      os << "?";
  }
  os << "]";
  return os;
}

loco::TensorShape own_shape(const luci::CircleNode *node)
{
  loco::TensorShape shape;
  shape.rank(node->rank());
  for (uint32_t r = 0; r < node->rank(); ++r)
  {
    // Shape inference rules in this file did not consider unknown dimension.
    // If some node has unknown dimension, 0 is inserted and wrong shape
    // inference was done as a result.
    // To fix this, new shape inference algorithm is being implemented.
    // Until new inference algorithm is fully implemented, unknown dimension
    // would be represented as 1 along with TFLite expression.
    shape.dim(r) = node->dim(r).known() ? node->dim(r).value() : 1;
  }
  return shape;
}

loco::NodeShape use_own(const luci::CircleNode *node)
{
  loco::TensorShape shape = own_shape(node);
  return loco::NodeShape{shape};
}

/**
 * @brief Create a higher-rank TensorShape following NumPy broadcasting semantics
 *
 * HOW TO USE:
 *
 *   auto expanded_tensor_shape = expand(tensor_shape).to(N);
 */
class TensorShapeExpander
{
public:
  TensorShapeExpander(const loco::TensorShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  loco::TensorShape to(uint32_t output_rank)
  {
    auto const &input_shape = _shape;
    uint32_t const input_rank = input_shape.rank();

    assert(input_rank <= output_rank && "Cannot shrink rank");
    uint32_t const axis_shift = output_rank - input_rank;

    loco::TensorShape output_shape;

    output_shape.rank(output_rank);
    for (uint32_t axis = 0; axis < output_rank; ++axis)
    {
      output_shape.dim(axis) = (axis < axis_shift) ? 1 : input_shape.dim(axis - axis_shift);
    }

    return output_shape;
  }

private:
  const loco::TensorShape _shape;
};

/**
 * @brief  Expand shape x and y to same rank by align right and filling with 1
 */
void expand_rank(loco::TensorShape &x, loco::TensorShape &y)
{
  auto x_rank = x.rank();
  auto y_rank = y.rank();

  if (x_rank == y_rank)
    return;

  TensorShapeExpander x_exp(x);
  TensorShapeExpander y_exp(y);

  auto xy_rank = std::max(x_rank, y_rank);

  x = x_rank > y_rank ? x : x_exp.to(xy_rank);
  y = y_rank > x_rank ? y : y_exp.to(xy_rank);
}

/**
 * @brief  Returns shape of expanded dimension of input x and y having same rank
 */
loco::TensorShape expand_dimension(const loco::TensorShape &x, const loco::TensorShape &y)
{
  assert(x.rank() == y.rank());

  auto rank = x.rank();

  loco::TensorShape output_shape;

  output_shape.rank(rank);
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    auto x_dim = x.dim(axis).known() ? x.dim(axis).value() : 1;
    auto y_dim = y.dim(axis).known() ? y.dim(axis).value() : 1;

    // each dimension of x and y should be same or one must be 1 if different
    if (!((x_dim == y_dim) || (x_dim == 1 || y_dim == 1)))
      INTERNAL_EXN("Cannot produce expand_dimension of two shapes");

    output_shape.dim(axis) = std::max(x_dim, y_dim);
  }

  return output_shape;
}

loco::TensorShape broadcast_shape(const loco::TensorShape &x, const loco::TensorShape &y)
{
  auto x_match = x;
  auto y_match = y;

  expand_rank(x_match, y_match);

  auto output_shape = expand_dimension(x_match, y_match);

  return output_shape;
}

/**
 * @brief vector_from_constant will return int64_t vector from CircleConst node
 */
template <loco::DataType T> std::vector<int64_t> vector_from_constant(luci::CircleConst *const_node)
{
  std::vector<int64_t> result;

  for (uint32_t idx = 0; idx < const_node->size<T>(); ++idx)
    result.push_back(const_node->at<T>(idx));

  return result;
}

template <class CIRCLENODE> loco::NodeShape broadcast_xy(const CIRCLENODE *node)
{
  auto x_shape = luci::shape_get(node->x()).template as<loco::TensorShape>();
  auto y_shape = luci::shape_get(node->y()).template as<loco::TensorShape>();

  auto output_shape = broadcast_shape(x_shape, y_shape);

  return loco::NodeShape{output_shape};
}

#define DECLARE_USE_SINGLE(NAME)                                                        \
  template <class CIRCLENODE> loco::NodeShape use_##NAME(const CIRCLENODE *node)        \
  {                                                                                     \
    auto inputs_shape = luci::shape_get(node->NAME()).template as<loco::TensorShape>(); \
    return loco::NodeShape{inputs_shape};                                               \
  }

DECLARE_USE_SINGLE(input);
DECLARE_USE_SINGLE(inputs);
DECLARE_USE_SINGLE(x);
DECLARE_USE_SINGLE(logits);

#undef DECLARE_USE_SINGLE

template <class CIRCLENODE>
loco::NodeShape use_paddings(const CIRCLENODE *node, const luci::CircleConst *paddings)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto input_shape = luci::shape_get(node->input()).template as<loco::TensorShape>();

  // TODO support other data type
  LUCI_ASSERT(paddings->dtype() == S32, "Only support int 32 for now");
  LUCI_ASSERT(paddings->rank() == 2, "paddings should be rank 2")

  int32_t n = paddings->dim(0).value();
  int32_t v = paddings->dim(1).value();

  LUCI_ASSERT(v == 2, "paddings should be [n, 2]");
  LUCI_ASSERT(n == int32_t(input_shape.rank()),
              "paddings [n, 2] should have same value of input rank");

  loco::TensorShape output_shape;

  output_shape.rank(input_shape.rank());
  for (int32_t ni = 0; ni < n; ++ni)
  {
    int32_t idx = ni * 2;
    int value = input_shape.dim(ni).value();
    value += paddings->at<S32>(idx + 0); // left
    value += paddings->at<S32>(idx + 1); // right
    output_shape.dim(ni) = value;
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_add_n(const luci::CircleAddN *node)
{
  auto shape = luci::shape_get(node->inputs(0)).as<loco::TensorShape>();

  for (uint32_t idx = 1; idx < node->arity(); ++idx)
  {
    auto shape_idx = luci::shape_get(node->inputs(idx)).as<loco::TensorShape>();
    if (!(shape == shape_idx))
    {
      INTERNAL_EXN_V("ADD_N shape not same as the first input: ", idx);
    }
  }
  return loco::NodeShape{shape};
}

template <class CIRCLENODE> loco::NodeShape infer_arg_maxmin(const CIRCLENODE *node)
{
  auto input_shape = luci::shape_get(node->input()).template as<loco::TensorShape>();
  auto dimension_shape = luci::shape_get(node->dimension()).template as<loco::TensorShape>();

  int64_t select_axis = 0;
  {
    LUCI_ASSERT(node->dimension(), "2nd input dimension() should not be nullptr");

    // Only support node's shape() is CircleConst with S32/S64
    // Support S32 for now.
    auto const_shape_node = loco::must_cast<luci::CircleConst *>(node->dimension());
    LUCI_ASSERT(const_shape_node->dtype() == loco::DataType::S32,
                "Only support int32 CircleConst for CircleArgMax/CircleArgMin");

    if (const_shape_node->rank() > 1)
      INTERNAL_EXN_V("Only support rank 0/1 CircleConst",
                     oops::to_uint32(const_shape_node->rank()));

    select_axis = const_shape_node->template scalar<loco::DataType::S32>();
  }

  assert(select_axis < input_shape.rank());

  if (select_axis < 0)
    select_axis += static_cast<int64_t>(input_shape.rank());

  // NOTE select_axis is removed
  loco::TensorShape shape_output;
  uint32_t rank = input_shape.rank();
  uint32_t shrink = static_cast<uint32_t>(select_axis);
  assert(rank > 0);
  shape_output.rank(rank - 1);
  for (uint32_t r = 0, d = 0; r < rank; ++r)
  {
    if (r == shrink)
      continue;
    shape_output.dim(d++) = input_shape.dim(r);
  }
  return loco::NodeShape{shape_output};
}

// Call this for CircleAvgPool2D and CircleMaxPool2D only
template <class Pool2DType> loco::NodeShape infer_pool_2d_shape(const Pool2DType *node)
{
  auto ifm_shape = luci::shape_get(node->value()).template as<loco::TensorShape>();
  assert(ifm_shape.rank() == 4);
  assert(ifm_shape.dim(1).known());
  assert(ifm_shape.dim(2).known());

  uint32_t input_height = ifm_shape.dim(1).value();
  uint32_t input_width = ifm_shape.dim(2).value();
  uint32_t stride_height = node->stride()->h();
  uint32_t stride_width = node->stride()->w();
  uint32_t window_height = node->filter()->h();
  uint32_t window_width = node->filter()->w();
  uint32_t dilation_height = 1; // dilation for CircleAvgPool2D and CircleMaxPool2D is 1
  uint32_t dilation_width = 1;
  uint32_t effective_window_height = dilation_height * (window_height - 1) + 1;
  uint32_t effective_window_width = dilation_width * (window_width - 1) + 1;

  uint32_t output_height = 0;
  uint32_t output_width = 0;

  if (node->padding() == luci::Padding::VALID)
  {
    LUCI_ASSERT(input_height + stride_height > effective_window_height, "Invalid shape");
    LUCI_ASSERT(input_width + stride_width > effective_window_width, "Invalid shape");
    output_height = (input_height + stride_height - effective_window_height) / stride_height;
    output_width = (input_width + stride_width - effective_window_width) / stride_width;
  }
  else if (node->padding() == luci::Padding::SAME)
  {
    output_height = (input_height + stride_height - 1) / stride_height;
    output_width = (input_width + stride_width - 1) / stride_width;
  }
  else
    LUCI_ASSERT(false, "Wrong padding type");

  loco::TensorShape ofm_shape;
  ofm_shape.rank(4);
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(1) = output_height;
  ofm_shape.dim(2) = output_width;
  ofm_shape.dim(3) = ifm_shape.dim(3);

  return loco::NodeShape{ofm_shape};
}

loco::NodeShape infer_batch_to_space_nd(const luci::CircleBatchToSpaceND *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  // Support only input rank is 3 and 4
  assert(input_shape.rank() == 3 || input_shape.rank() == 4);

  // Only support block_shape() with S32 type CircleConst for now
  auto const_block_shape = loco::must_cast<luci::CircleConst *>(node->block_shape());
  LUCI_ASSERT(const_block_shape->dtype() == loco::DataType::S32, "Only support int32 block_shape");

  // Only support crops() with S32 type CircleConst for now
  auto const_crops = loco::must_cast<luci::CircleConst *>(node->crops());
  LUCI_ASSERT(const_crops->dtype() == loco::DataType::S32, "Only support int32 crops");

  auto const_block_shape_shape = luci::shape_get(const_block_shape).as<loco::TensorShape>();
  auto const_crops_shape = luci::shape_get(const_crops).as<loco::TensorShape>();
  assert(const_block_shape_shape.rank() == 1);
  assert(const_crops_shape.rank() == 2);

  int32_t input_spatial_dim = input_shape.rank() - 2;
  assert(const_block_shape_shape.dim(0) == input_spatial_dim);
  assert(const_crops_shape.dim(0) == input_spatial_dim);
  assert(const_crops_shape.dim(1) == 2);

  loco::TensorShape shape_output;

  shape_output.rank(input_shape.rank());

  int32_t output_batch_size = input_shape.dim(0).value();
  for (int32_t dim = 0; dim < input_spatial_dim; ++dim)
  {
    int dim_size = input_shape.dim(dim + 1).value() * const_block_shape->at<S32>(dim);
    dim_size -= const_crops->at<S32>(dim * 2);
    dim_size -= const_crops->at<S32>(dim * 2 + 1);
    shape_output.dim(dim + 1) = dim_size;

    assert(output_batch_size % const_block_shape->at<S32>(dim) == 0);
    output_batch_size = output_batch_size / const_block_shape->at<S32>(dim);
  }
  shape_output.dim(0) = output_batch_size;
  shape_output.dim(input_shape.rank() - 1) = input_shape.dim(input_shape.rank() - 1);

  return loco::NodeShape{shape_output};
}

struct OutputSize
{
  uint32_t height = 0;
  uint32_t width = 0;
};

template <class Conv2DType> OutputSize infer_conv2d_type(const Conv2DType *node)
{
  auto ifm_shape = luci::shape_get(node->input()).template as<loco::TensorShape>();
  auto ker_shape = luci::shape_get(node->filter()).template as<loco::TensorShape>();
  assert(ifm_shape.rank() == 4);
  assert(ker_shape.rank() == 4);
  assert(ifm_shape.dim(1).known());
  assert(ifm_shape.dim(2).known());
  assert(ker_shape.dim(1).known());
  assert(ker_shape.dim(2).known());

  uint32_t input_height = ifm_shape.dim(1).value();
  uint32_t input_width = ifm_shape.dim(2).value();
  uint32_t stride_height = node->stride()->h();
  uint32_t stride_width = node->stride()->w();
  uint32_t ker_height = ker_shape.dim(1).value();
  uint32_t ker_width = ker_shape.dim(2).value();
  uint32_t dilation_height = node->dilation()->h();
  uint32_t dilation_width = node->dilation()->w();
  uint32_t effective_ker_height = dilation_height * (ker_height - 1) + 1;
  uint32_t effective_ker_width = dilation_width * (ker_width - 1) + 1;

  uint32_t output_height = 0;
  uint32_t output_width = 0;

  if (node->padding() == luci::Padding::VALID)
  {
    LUCI_ASSERT(input_height + stride_height > effective_ker_height, "Invalid shape");
    LUCI_ASSERT(input_width + stride_width > effective_ker_width, "Invalid shape");
    output_height = (input_height + stride_height - effective_ker_height) / stride_height;
    output_width = (input_width + stride_width - effective_ker_width) / stride_width;
  }
  else if (node->padding() == luci::Padding::SAME)
  {
    output_height = (input_height + stride_height - 1) / stride_height;
    output_width = (input_width + stride_width - 1) / stride_width;
  }
  else
    LUCI_ASSERT(false, "Wrong padding type");

  OutputSize os{output_height, output_width};

  return os;
}

// BatchMatMulV2 supports broadcasting in the batch dimensions(BatchMatMul doesn't)
// TODO Distinguish BatchMatMul and BatchMatMulV2
loco::NodeShape infer_batchmatmul_shape(const loco::TensorShape &x_shape,
                                        const loco::TensorShape &y_shape, bool adj_x, bool adj_y)
{
  uint32_t x_rank = x_shape.rank();
  uint32_t y_rank = y_shape.rank();
  assert(x_rank >= 2 && y_rank >= 2);

  loco::TensorShape output_shape;
  output_shape.rank(x_shape.rank());
  // Braodcast in the batch dimension
  if (x_rank > 2 || y_rank > 2)
  {
    loco::TensorShape dummy_x = x_shape;
    loco::TensorShape dummy_y = y_shape;
    expand_rank(dummy_x, dummy_y);
    if (x_rank < y_rank)
      expand_rank(output_shape, dummy_y);

    for (uint32_t d = 0; d < output_shape.rank() - 2; d++)
    {
      uint32_t max_dim = std::max(dummy_x.dim(d).value(), dummy_y.dim(d).value());
      if (dummy_x.dim(d) == dummy_y.dim(d) ||
          dummy_x.dim(d).value() * dummy_y.dim(d).value() == max_dim)
        output_shape.dim(d).set(max_dim);
      else
        INTERNAL_EXN("BatchMatMul has wrong shape");
    }
  }

  loco::Dimension x_lhs = adj_x ? x_shape.dim(x_rank - 1) : x_shape.dim(x_rank - 2);
  loco::Dimension x_rhs = adj_x ? x_shape.dim(x_rank - 2) : x_shape.dim(x_rank - 1);
  loco::Dimension y_lhs = adj_y ? y_shape.dim(y_rank - 1) : y_shape.dim(y_rank - 2);
  loco::Dimension y_rhs = adj_y ? y_shape.dim(y_rank - 2) : y_shape.dim(y_rank - 1);

  if (x_rhs.known() && y_lhs.known() && not(x_rhs == y_lhs))
    INTERNAL_EXN("x_rhs and y_lhs should be same");

  uint32_t out_rank = output_shape.rank();
  output_shape.dim(out_rank - 2) = x_lhs;
  output_shape.dim(out_rank - 1) = y_rhs;

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_broadcast_to(const luci::CircleBroadcastTo *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  loco::TensorShape shape_by_input;
  {
    LUCI_ASSERT(node->shape(), "2nd input shape() should not be nullptr");

    // Only support node's shape() is CircleConst with S32
    auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->shape());
    if (const_shape_node != nullptr)
    {
      LUCI_ASSERT(const_shape_node->dtype() == S32, "Only support int32 CircleConst");

      shape_by_input.rank(const_shape_node->size<S32>());
      for (uint32_t axis = 0; axis < shape_by_input.rank(); ++axis)
      {
        shape_by_input.dim(axis) = const_shape_node->at<S32>(axis);
      }
    }
    else
    {
      // We use shape from the node itself
      shape_by_input = own_shape(node);
    }
  }

  return loco::NodeShape{shape_by_input};
}

loco::NodeShape infer_concatenation(const luci::CircleConcatenation *node)
{
  // TODO Support when CircleConcatenation has 0 input
  assert(node->numValues() > 0);

  auto first_shape = luci::shape_get(node->values(0)).as<loco::TensorShape>();
  auto axis = node->axis();
  if (axis < 0)
    axis += first_shape.rank();

  assert(0 <= axis);
  assert(first_shape.rank() > static_cast<uint32_t>(axis));

  loco::TensorShape output_shape;

  output_shape.rank(first_shape.rank());
  for (uint32_t i = 0; i < output_shape.rank(); ++i)
    output_shape.dim(i) = first_shape.dim(i);

  for (uint32_t i = 1; i < node->numValues(); ++i)
  {
    auto input_shape = luci::shape_get(node->values(i)).as<loco::TensorShape>();

    for (uint32_t j = 0; j < output_shape.rank(); ++j)
    {
      if (j == static_cast<uint32_t>(axis))
      {
        // If dimension is unknown, value() will return 0.
        // This is wrong but until new inference algorithm is implemented,
        // this code will not be modified to keep compatibility.
        output_shape.dim(j) = output_shape.dim(j).value() + input_shape.dim(j).value();
      }
      else
        assert(!output_shape.dim(j).known() || !input_shape.dim(j).known() ||
               output_shape.dim(j) == input_shape.dim(j));
    }
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_conv2d(const luci::CircleConv2D *node)
{
  LOGGER(l);

  auto ifm_shape = luci::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
  auto ker_shape = luci::shape_get(node->filter()).as<loco::TensorShape>(); // in OHWI

  assert(ifm_shape.rank() == 4);
  assert(ker_shape.rank() == 4);
  assert(ifm_shape.dim(3) == ker_shape.dim(3));

  auto os = infer_conv2d_type(node);

  loco::TensorShape ofm_shape;
  ofm_shape.rank(4);
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(1) = os.height;
  ofm_shape.dim(2) = os.width;
  ofm_shape.dim(3) = ker_shape.dim(0);

  INFO(l) << "[luci] CircleConv2D ShapeInf ifm(" << ifm_shape.rank() << ") ker(" << ker_shape.rank()
          << ") output(" << ofm_shape.dim(0).value() << "," << ofm_shape.dim(1).value() << ","
          << ofm_shape.dim(2).value() << "," << ofm_shape.dim(3).value() << ") " << node->name()
          << std::endl;

  return loco::NodeShape{ofm_shape};
}

loco::NodeShape infer_depth_to_space(const luci::CircleDepthToSpace *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  LUCI_ASSERT(input_shape.rank() == 4, "Only input rank 4 is supported");

  // Only data format NHWC is supported
  // TODO need to clarify what to do with layout in this operator
  int32_t height = input_shape.dim(1).value();
  int32_t width = input_shape.dim(2).value();
  int32_t depth = input_shape.dim(3).value();

  int block_size = node->block_size();

  if (block_size < 2)
    INTERNAL_EXN("Block size must be >= 2");

  if (depth % (block_size * block_size))
  {
    INTERNAL_EXN("The input tensor's depth must be divisible by block_size^2");
  }

  loco::TensorShape output_shape;
  output_shape.rank(4);

  output_shape.dim(0) = input_shape.dim(0).value();
  output_shape.dim(1) = height * block_size;
  output_shape.dim(2) = width * block_size;
  output_shape.dim(3) = depth / (block_size * block_size);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_depthwise_conv2d(const luci::CircleDepthwiseConv2D *node)
{
  auto ifm_shape = luci::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
  auto ker_shape = luci::shape_get(node->filter()).as<loco::TensorShape>(); // in 1 H W CM

  assert(ifm_shape.rank() == 4);
  assert(ker_shape.rank() == 4);
  assert(ker_shape.dim(0).value() == 1);
  assert(ifm_shape.dim(3).value() * node->depthMultiplier() == ker_shape.dim(3).value());

  auto os = infer_conv2d_type(node);

  loco::TensorShape ofm_shape;
  ofm_shape.rank(4);
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(1) = os.height;
  ofm_shape.dim(2) = os.width;
  ofm_shape.dim(3) = ker_shape.dim(3);

  return loco::NodeShape{ofm_shape};
}

loco::NodeShape infer_expand_dims(const luci::CircleExpandDims *node)
{
  const loco::DataType S32 = loco::DataType::S32;
  auto x_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  if (x_shape.rank() == 0)
  {
    // This maybe for unknown shape. We use shape from the node itself.
    return use_own(node);
  }
  auto const_axis = loco::must_cast<luci::CircleConst *>(node->axis());
  LUCI_ASSERT(const_axis->dtype() == S32, "Only support int32 CircleConst for axis");
  if (const_axis->rank() != 0 && const_axis->rank() != 1)
  {
    INTERNAL_EXN_V("Non-scalar axis in OP", node->opnum());
  }
  int32_t axis = const_axis->at<S32>(0);
  LUCI_ASSERT((axis <= static_cast<int32_t>(x_shape.rank())) &&
                (axis >= -1 - static_cast<int32_t>(x_shape.rank())),
              "Axis has to be between [-(D+1), D], where D is rank of input.");
  size_t positive_axis = axis < 0 ? x_shape.rank() + axis + 1 : axis;
  loco::TensorShape output_shape;
  output_shape.rank(x_shape.rank() + 1);
  size_t i = 0;
  for (; i < positive_axis; i++)
    output_shape.dim(i) = x_shape.dim(i);
  output_shape.dim(i) = loco::Dimension(1);
  for (; i < x_shape.rank(); i++)
    output_shape.dim(i + 1) = x_shape.dim(i);
  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_fill(const luci::CircleFill *node)
{
  loco::TensorShape shape;
  {
    LUCI_ASSERT(node->dims(), "dims input should not be nullptr");

    auto dims_node = dynamic_cast<luci::CircleConst *>(node->dims());
    if (dims_node != nullptr)
    {
      // Only support node with S32
      LUCI_ASSERT(dims_node->dtype() == loco::DataType::S32, "Only support int32 CircleConst");

      if (dims_node->rank() != 1)
        INTERNAL_EXN_V("Only support rank 1 CircleConst", oops::to_uint32(dims_node->rank()));

      shape.rank(dims_node->dim(0).value());

      for (uint32_t axis = 0; axis < shape.rank(); ++axis)
      {
        shape.dim(axis) = dims_node->at<loco::DataType::S32>(axis);
      }
    }
    else
    {
      shape = own_shape(node);
    }
  }

  return loco::NodeShape{shape};
}

loco::NodeShape infer_fully_connected(const luci::CircleFullyConnected *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto weights_shape = luci::shape_get(node->weights()).as<loco::TensorShape>();

  loco::TensorShape out_shape;

  // NOTE Some recipes in some repositories are using rank 4 input for FullyConnected.
  //      Until they are all fixed, disable following assert.
  // TODO Enable following assert after related fixes are applied
  // https://github.com/tensorflow/tensorflow/blob/ea33c1e7a25d8025e8ee405ad8ab7be261798d76/tensorflow/lite/kernels/fully_connected.cc#L194
  // LUCI_ASSERT(input_shape.rank() == 2 || input_shape.rank() == 3,
  //             "Input rank of FullyConnected should be 2 or 3");

  // https://github.com/tensorflow/tensorflow/blob/ea33c1e7a25d8025e8ee405ad8ab7be261798d76/tensorflow/lite/kernels/fully_connected.cc#L225
  LUCI_ASSERT(weights_shape.rank() == 2, "Weights of FullyConnected should be 2");

  // https://github.com/tensorflow/tensorflow/blob/ea33c1e7a25d8025e8ee405ad8ab7be261798d76/tensorflow/lite/kernels/fully_connected.cc#L353-L367
  if (node->keep_num_dims())
  {
    out_shape.rank(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i)
      out_shape.dim(i) = input_shape.dim(i);
    out_shape.dim(out_shape.rank() - 1) = weights_shape.dim(0);
  }
  else
  {
    uint32_t input_size = 1;
    for (uint32_t i = 0; i < input_shape.rank(); i++)
    {
      input_size = input_size * input_shape.dim(i).value();
    }
    const uint32_t batch_size = input_size / weights_shape.dim(1).value();
    out_shape.rank(2);
    out_shape.dim(0) = batch_size;
    out_shape.dim(1) = weights_shape.dim(0);
  }

  return loco::NodeShape{out_shape};
}

loco::NodeShape infer_gather(const luci::CircleGather *node)
{
  loco::TensorShape output_shape;

  const auto input_shape = luci::shape_get(node->params()).as<loco::TensorShape>();
  const auto positions_shape = luci::shape_get(node->indices()).as<loco::TensorShape>();
  int32_t axis = node->axis();

  // If CircleGather input has a dynamic shape, it can't inference this shape. So, it returns the
  // shape that node already has.
  if (input_shape.rank() == 0 || positions_shape.rank() == 0)
    return use_own(node);

  if (axis < 0)
    axis += input_shape.rank();

  output_shape.rank(input_shape.rank() - 1 + positions_shape.rank());
  int32_t outdim_index = 0;
  for (int32_t i = 0; i < axis; ++i)
    output_shape.dim(outdim_index++) = input_shape.dim(i);
  for (uint32_t i = 0; i < positions_shape.rank(); ++i)
    output_shape.dim(outdim_index++) = positions_shape.dim(i);
  for (uint32_t i = axis + 1; i < input_shape.rank(); ++i)
    output_shape.dim(outdim_index++) = input_shape.dim(i);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_gather_nd(const luci::CircleGatherNd *node)
{
  loco::TensorShape output_shape;

  const auto params_shape = luci::shape_get(node->params()).as<loco::TensorShape>();
  const auto indices_shape = luci::shape_get(node->indices()).as<loco::TensorShape>();

  const auto params_rank = params_shape.rank();
  const auto indices_rank = indices_shape.rank();

  // see https://www.tensorflow.org/api_docs/python/tf/gather_nd
  // output.shape = indices.shape[:-1] + params.shape[indices.shape[-1]:]
  // batch_dims isn't supported in tflite

  // TODO: replace exceptions with setting shape to unknown?

  if (!indices_shape.dim(indices_rank - 1).known())
    INTERNAL_EXN("Last indices dimension is unknown");

  auto indices_last_dim = indices_shape.dim(indices_rank - 1).value();

  if (indices_last_dim > params_rank)
    INTERNAL_EXN("Last indices dimension should be <= params rank");

  const uint32_t output_rank = indices_rank + params_rank - indices_last_dim - 1;

  output_shape.rank(output_rank);

  uint32_t output_index = 0;
  for (uint32_t i = 0; i < indices_rank - 1; ++i)
  {
    auto &dim = indices_shape.dim(i);
    if (!dim.known())
      INTERNAL_EXN("Unknown indices dimension is unsupported");
    output_shape.dim(output_index++).set(dim.value());
  }

  for (uint32_t i = indices_last_dim; i < params_rank; ++i)
  {
    auto &dim = params_shape.dim(i);
    if (!dim.known())
      INTERNAL_EXN("Unknown params dimension is unsupported");
    output_shape.dim(output_index++).set(dim.value());
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_matrix_diag(const luci::CircleMatrixDiag *node)
{
  loco::TensorShape output_shape;

  auto diagonal_shape = luci::shape_get(node->diagonal()).as<loco::TensorShape>();
  auto rank = diagonal_shape.rank();

  output_shape.rank(rank + 1);

  for (uint32_t i = 0; i < rank; i++)
  {
    output_shape.dim(i) = diagonal_shape.dim(i);
  }

  output_shape.dim(rank) = diagonal_shape.dim(rank - 1);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_matrix_set_diag(const luci::CircleMatrixSetDiag *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto diagonal_shape = luci::shape_get(node->diagonal()).as<loco::TensorShape>();

  auto rank = diagonal_shape.rank();

  LUCI_ASSERT(rank == input_shape.rank() - 1, "diagonal rank = input rank - 1");

  for (uint32_t i = 0; i < rank - 1; i++)
  {
    LUCI_ASSERT(diagonal_shape.dim(i) == input_shape.dim(i), "diagonal dims = input dims");
  }

  auto dim = std::min(input_shape.dim(rank - 1).value(), input_shape.dim(rank).value());

  LUCI_ASSERT(dim == diagonal_shape.dim(rank - 1), "Max diag len error");

  return loco::NodeShape{input_shape};
}

loco::TensorShape infer_reducer(const loco::Node *input, const loco::Node *indices, bool keep_dims)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto input_shape = luci::shape_get(input).as<loco::TensorShape>();
  auto reduction_indices = loco::must_cast<const luci::CircleConst *>(indices);

  { // Exceptions
    // TODO support non-const case
    // TODO support other data type
    LUCI_ASSERT(reduction_indices->dtype() == S32, "Only support int 32");
  }

  std::vector<int32_t> reduction_values;

  for (uint32_t i = 0; i < reduction_indices->size<S32>(); ++i)
  {
    int32_t axis = reduction_indices->at<S32>(i);
    if (axis < 0)
      axis += input_shape.rank();
    if (not(0 <= axis and axis < static_cast<int32_t>(input_shape.rank())))
      INTERNAL_EXN_V("Invalid reduction axis for REDUCER", oops::to_uint32(axis));
    reduction_values.push_back(axis);
  }

  loco::TensorShape output_shape;

  if (keep_dims)
  {
    output_shape.rank(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i)
      output_shape.dim(i) = input_shape.dim(i);
    for (uint32_t i = 0; i < reduction_values.size(); ++i)
      output_shape.dim(reduction_values.at(i)) = 1;
  }
  else
  {
    std::vector<bool> check_reduce(input_shape.rank(), false);
    for (uint32_t i = 0; i < reduction_values.size(); ++i)
      check_reduce.at(reduction_values.at(i)) = true;

    uint32_t reduce_cnt = 0;
    for (uint32_t i = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i))
        ++reduce_cnt;

    output_shape.rank(input_shape.rank() - reduce_cnt);
    for (uint32_t i = 0, j = 0; i < check_reduce.size(); ++i)
      if (check_reduce.at(i) == false)
        output_shape.dim(j++) = input_shape.dim(i);
  }

  return output_shape;
}

loco::NodeShape infer_mirror_pad(const luci::CircleMirrorPad *node)
{
  // TODO support non-const case
  auto paddings = loco::must_cast<luci::CircleConst *>(node->paddings());
  return use_paddings(node, paddings);
}

loco::NodeShape infer_one_hot(const luci::CircleOneHot *node)
{
  const loco::DataType S32 = loco::DataType::S32;
  auto indices_shape = luci::shape_get(node->indices()).as<loco::TensorShape>();
  // Only support OneHot node's depth() is CircleConst with type S32
  // TODO support depth with other types
  auto depth = loco::must_cast<luci::CircleConst *>(node->depth());
  LUCI_ASSERT(depth->dtype() == S32, "Only support int32 CircleConst");
  if (depth->rank() != 0)
    INTERNAL_EXN_V("Only support rank 0 CircleOneHot in Depth", oops::to_uint32(depth->rank()));
  loco::TensorShape output_shape;
  output_shape.rank(indices_shape.rank() + 1);
  auto axis = node->axis();
  if (axis < 0)
    axis += indices_shape.rank() + 1;
  LUCI_ASSERT(0 <= axis, "Axis is out of range");
  LUCI_ASSERT(static_cast<uint32_t>(axis) <= indices_shape.rank(), "Axis is out of range");
  uint32_t j = 0;
  for (uint32_t i = 0; i < output_shape.rank(); i++)
  {
    if (i == static_cast<uint32_t>(axis))
    {
      output_shape.dim(i) = depth->at<S32>(0);
    }
    else
    {
      output_shape.dim(i) = indices_shape.dim(j++);
    }
  }
  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_pack(const luci::CirclePack *node)
{
  LUCI_ASSERT(node->values_count() > 0, "Only support one or more inputs");

  auto first_shape = luci::shape_get(node->values(0)).as<loco::TensorShape>();
  // Make sure all inputs have the same shape.
  for (uint32_t i = 1; i < node->values_count(); ++i)
  {
    auto in_shape = luci::shape_get(node->values(i)).as<loco::TensorShape>();
    LUCI_ASSERT(loco::NodeShape{first_shape} == loco::NodeShape{in_shape},
                "All inputs must have the same shape");
  }

  // Checking shape capability for pack layer
  // Input: tensors [D1, D2, ... Dn]
  // Axis: K
  // Output: [D1, D2, ... , D_K-1, n, D_K+1, ... Dn]
  auto axis = node->axis();
  if (axis < 0)
    axis += first_shape.rank() + 1;

  LUCI_ASSERT(0 <= axis, "Axis is out of range");
  LUCI_ASSERT(static_cast<uint32_t>(axis) <= first_shape.rank(), "Axis is out of range");

  loco::TensorShape output_shape;
  output_shape.rank(first_shape.rank() + 1);

  uint32_t j = 0;
  for (uint32_t i = 0; i < output_shape.rank(); ++i)
  {
    if (i == static_cast<uint32_t>(axis))
    {
      output_shape.dim(i) = node->values_count();
    }
    else
    {
      output_shape.dim(i) = first_shape.dim(j++);
    }
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_pad(const luci::CirclePad *node)
{
  // TODO support non-const case
  auto paddings = loco::must_cast<luci::CircleConst *>(node->paddings());
  return use_paddings(node, paddings);
}

loco::NodeShape infer_pad_v2(const luci::CirclePadV2 *node)
{
  // TODO support non-const case
  auto paddings = dynamic_cast<luci::CircleConst *>(node->paddings());
  if (!paddings)
  {
    auto node_shape = own_shape(node);
    return loco::NodeShape{node_shape};
  }
  return use_paddings(node, paddings);
}

loco::NodeShape infer_p_relu(const luci::CirclePRelu *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto alpha_shape = luci::shape_get(node->alpha()).as<loco::TensorShape>();

  auto output_shape = broadcast_shape(input_shape, alpha_shape);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_range(const luci::CircleRange *node)
{
  loco::TensorShape output_shape;
  output_shape.rank(1);

  auto start_node = dynamic_cast<luci::CircleConst *>(node->start());
  auto limit_node = dynamic_cast<luci::CircleConst *>(node->limit());
  auto delta_node = dynamic_cast<luci::CircleConst *>(node->delta());

  if (start_node == nullptr || limit_node == nullptr || delta_node == nullptr)
  {
    return use_own(node);
  }

  double start = 0, limit = 0, delta = 0;

#define GET_RANGE_PARAM(DT)         \
  start = start_node->scalar<DT>(); \
  limit = limit_node->scalar<DT>(); \
  delta = delta_node->scalar<DT>();

  switch (start_node->dtype())
  {
    case loco::DataType::FLOAT32:
      GET_RANGE_PARAM(loco::DataType::FLOAT32)
      break;
    case loco::DataType::S32:
      GET_RANGE_PARAM(loco::DataType::S32)
      break;
    default:
      INTERNAL_EXN("Range data type not supported");
  }

#undef GET_RANGE_PARAM

  if (delta == 0)
    INTERNAL_EXN("Delta can not be zero");

  output_shape.dim(0) = ceil((limit - start) / delta);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_reshape(const luci::CircleReshape *node)
{
  LOGGER(l);

  const loco::DataType S32 = loco::DataType::S32;

  loco::TensorShape shape_by_input;
  {
    LUCI_ASSERT(node->shape(), "2nd input shape() should not be nullptr");

    // Only support node's shape() is CircleConst with S32
    // TODO support other node with other types
    auto const_shape_node = dynamic_cast<luci::CircleConst *>(node->shape());
    if (const_shape_node != nullptr)
    {
      LUCI_ASSERT(const_shape_node->dtype() == S32, "Only support int32 CircleConst");

      shape_by_input.rank(const_shape_node->size<S32>());

      for (uint32_t axis = 0; axis < shape_by_input.rank(); ++axis)
      {
        shape_by_input.dim(axis) = const_shape_node->at<S32>(axis);
      }
    }
    else
    {
      // We use shape from the node itself
      shape_by_input = own_shape(node);
    }
  }

  loco::TensorShape shape_by_attr;
  {
    shape_by_attr.rank(node->newShape()->rank());

    for (uint32_t axis = 0; axis < shape_by_attr.rank(); ++axis)
    {
      shape_by_attr.dim(axis) = node->newShape()->dim(axis);
    }
  }

  if (!(shape_by_input == shape_by_attr))
  {
    INFO(l) << "CircleReshape: Two new shape information mismatched : " << std::endl;
    INFO(l) << "   shape_by_input : " << shape_by_input << std::endl;
    INFO(l) << "   shape_by_attr : " << shape_by_attr << std::endl;
  }

  loco::TensorShape output_shape = shape_by_input;

  // One of the dimensions can have special value -1, meaning its actual value should be inferred.
  const auto input_shape = luci::shape_get(node->tensor()).as<loco::TensorShape>();
  uint32_t input_element_count = 1;
  uint32_t output_element_count = 1;
  uint32_t unknown_dim_index = UINT32_MAX;
  for (uint32_t i = 0; i < input_shape.rank(); ++i)
    input_element_count *= (input_shape.dim(i).known() ? input_shape.dim(i).value() : 1);
  for (uint32_t dim_index = 0; dim_index < output_shape.rank(); ++dim_index)
  {
    const uint32_t dim_value = output_shape.dim(dim_index).value();
    if (static_cast<int>(dim_value) == -1)
    {
      LUCI_ASSERT(unknown_dim_index == UINT32_MAX, "More than one unknown dimension");
      unknown_dim_index = dim_index;
    }
    else
    {
      output_element_count *= dim_value;
    }
  }
  if (unknown_dim_index != UINT32_MAX)
  {
    output_shape.dim(unknown_dim_index) = input_element_count / output_element_count;
  }

  return loco::NodeShape{output_shape};
}

template <class CIRCLENODE> loco::NodeShape infer_resize_type(const CIRCLENODE *node)
{
  auto input_shape = luci::shape_get(node->input()).template as<loco::TensorShape>();

  if (input_shape.rank() != 4)
    INTERNAL_EXN("Expected input to have rank 4");

  auto *const_node = loco::must_cast<luci::CircleConst *>(node->size());

  if (const_node->dtype() != loco::DataType::S32)
    INTERNAL_EXN("Only S32 datatype is supported for size");

  if (const_node->rank() != 1)
    INTERNAL_EXN("Expected size tensor of rank 1");

  if (const_node->dim(0).value() != 2)
    INTERNAL_EXN("Expected size tensor with shape [2]");

  loco::TensorShape output_shape;
  output_shape.rank(4);
  output_shape.dim(0) = input_shape.dim(0);
  output_shape.dim(1) = const_node->template at<loco::DataType::S32>(0);
  output_shape.dim(2) = const_node->template at<loco::DataType::S32>(1);
  output_shape.dim(3) = input_shape.dim(3);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_scatter_nd(const luci::CircleScatterNd *node)
{
  loco::TensorShape output_shape;

  auto shape_node = loco::must_cast<luci::CircleConst *>(node->shape());

  const loco::DataType S32 = loco::DataType::S32;
  const loco::DataType S64 = loco::DataType::S64;

  std::vector<int64_t> vect_shape;

  if (shape_node->dtype() == S32)
    vect_shape = vector_from_constant<S32>(shape_node);
  else if (shape_node->dtype() == S64)
    vect_shape = vector_from_constant<S64>(shape_node);
  else
    LUCI_ASSERT(false, "Only support int32/int64 for shape()");

  output_shape.rank(vect_shape.size());
  for (uint32_t i = 0; i < vect_shape.size(); ++i)
    output_shape.dim(i) = vect_shape[i];

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_segment_sum(const luci::CircleSegmentSum *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto segment_shape = luci::shape_get(node->segment_ids()).as<loco::TensorShape>();

  LUCI_ASSERT(segment_shape.rank() == 1, "segment_ids must be 1-D tensor");
  LUCI_ASSERT(segment_shape.dim(0).value() == input_shape.dim(0).value(),
              "segment_ids size must be equal to the size of data's first dimension");

  auto ids_shape_value = loco::must_cast<luci::CircleConst *>(node->segment_ids());

  std::vector<int64_t> vect_ids;

  if (ids_shape_value->dtype() == loco::DataType::S32)
    vect_ids = vector_from_constant<loco::DataType::S32>(ids_shape_value);

  LUCI_ASSERT(std::is_sorted(vect_ids.begin(), vect_ids.end()),
              "segment_ids values should be sorted")

  loco::TensorShape output_shape;

  output_shape.rank(input_shape.rank());

  for (uint32_t i = 1; i < input_shape.rank(); ++i)
    output_shape.dim(i) = input_shape.dim(i);

  output_shape.dim(0) = vect_ids.back() + 1;

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_select(const luci::CircleSelect *node)
{
  auto t_shape = luci::shape_get(node->t()).as<loco::TensorShape>();
  assert(t_shape == luci::shape_get(node->e()).as<loco::TensorShape>());

  // condition shape validation
  auto c_shape = luci::shape_get(node->condition()).as<loco::TensorShape>();
  if (c_shape.rank() != t_shape.rank())
  {
    if (c_shape.rank() != 0 && c_shape.rank() != 1)
      INTERNAL_EXN_V("CircleSelect condition rank is not 0 nor 1: ", c_shape.rank());

    if (c_shape.rank() == 1)
    {
      if (c_shape.dim(0).value() != t_shape.dim(0).value())
        INTERNAL_EXN("CircleSelect condition dim(0) should match with t.dim(0)");
    }
  }

  return loco::NodeShape{t_shape};
}

loco::NodeShape infer_select_v2(const luci::CircleSelectV2 *node)
{
  auto c_shape = luci::shape_get(node->condition()).as<loco::TensorShape>();
  auto t_shape = luci::shape_get(node->t()).as<loco::TensorShape>();
  auto e_shape = luci::shape_get(node->e()).as<loco::TensorShape>();

  // validate ability to broadcast shapes to each other
  auto b_shape = broadcast_shape(broadcast_shape(c_shape, t_shape), e_shape);
  return loco::NodeShape{b_shape};
}

loco::NodeShape infer_shape(const luci::CircleShape *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

  loco::TensorShape output_shape;

  output_shape.rank(1);
  output_shape.dim(0) = input_shape.rank();

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_slice(const luci::CircleSlice *node)
{
  const loco::DataType S32 = loco::DataType::S32;
  const loco::DataType S64 = loco::DataType::S64;

  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

  auto const_begin = loco::must_cast<luci::CircleConst *>(node->begin());
  auto const_size = loco::must_cast<luci::CircleConst *>(node->size());

  loco::TensorShape output_shape;
  std::vector<int64_t> vect_begin; // to hold both S32/S64, we use int64_t
  std::vector<int64_t> vect_size;

  if (const_begin->dtype() == S32)
    vect_begin = vector_from_constant<S32>(const_begin);
  else if (const_begin->dtype() == S64)
    vect_begin = vector_from_constant<S64>(const_begin);
  else
    LUCI_ASSERT(false, "Only support int32/int64 for begin()");

  if (const_size->dtype() == S32)
    vect_size = vector_from_constant<S32>(const_size);
  else if (const_size->dtype() == S64)
    vect_size = vector_from_constant<S64>(const_size);
  else
    LUCI_ASSERT(false, "Only support int32/int64 for size()");

  assert(input_shape.rank() == vect_begin.size());
  assert(input_shape.rank() == vect_size.size());

  output_shape.rank(vect_begin.size());
  for (uint32_t idx = 0; idx < vect_begin.size(); ++idx)
  {
    auto size = vect_size.at(idx);
    if (size == -1)
    {
      size = static_cast<int64_t>(input_shape.dim(idx).value()) - vect_begin.at(idx);
    }
    output_shape.dim(idx) = size;
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_space_to_batch_nd(const luci::CircleSpaceToBatchND *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  // Support only input rank is 3 and 4
  assert(input_shape.rank() == 3 || input_shape.rank() == 4);

  // Only support block_shape() with S32 type CircleConst for now
  auto const_block_shape = loco::must_cast<luci::CircleConst *>(node->block_shape());
  LUCI_ASSERT(const_block_shape->dtype() == S32, "Only support int32 block_shape");

  // Only support paddings() with S32 type CircleConst for now
  auto const_paddings = loco::must_cast<luci::CircleConst *>(node->paddings());
  LUCI_ASSERT(const_paddings->dtype() == S32, "Only support int32 paddings");

  auto const_block_shape_shape = luci::shape_get(const_block_shape).as<loco::TensorShape>();
  auto const_paddings_shape = luci::shape_get(const_paddings).as<loco::TensorShape>();
  assert(const_block_shape_shape.rank() == 1);
  assert(const_paddings_shape.rank() == 2);

  int32_t input_spatial_dim = input_shape.rank() - 2;
  assert(const_block_shape_shape.dim(0) == input_spatial_dim);
  assert(const_paddings_shape.dim(0) == input_spatial_dim);
  assert(const_paddings_shape.dim(1) == 2);

  // Check all values of block_shape >= 1
  uint32_t ele_count = const_block_shape->size<S32>();
  for (uint32_t e = 0; e < ele_count; ++e)
  {
    auto val = const_block_shape->at<S32>(e);
    if (val < 1)
    {
      INTERNAL_EXN_V("All values of block_shape >= 1: ", e);
    }
  }

  loco::TensorShape shape_output;

  shape_output.rank(input_shape.rank());

  int32_t output_batch_size = input_shape.dim(0).value();
  for (int32_t dim = 0; dim < input_spatial_dim; ++dim)
  {
    int dim_size = input_shape.dim(dim + 1).value();
    dim_size += const_paddings->at<S32>(dim * 2);
    dim_size += const_paddings->at<S32>(dim * 2 + 1);
    shape_output.dim(dim + 1) = dim_size / const_block_shape->at<S32>(dim);

    assert(dim_size % const_block_shape->at<S32>(dim) == 0);
    output_batch_size = output_batch_size * const_block_shape->at<S32>(dim);
  }
  shape_output.dim(0) = output_batch_size;
  shape_output.dim(input_shape.rank() - 1) = input_shape.dim(input_shape.rank() - 1);

  return loco::NodeShape{shape_output};
}

loco::NodeShape infer_space_to_depth(const luci::CircleSpaceToDepth *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  LUCI_ASSERT(input_shape.rank() == 4, "Only input rank 4 is supported");

  // Only data format NHWC is supported
  int32_t height = input_shape.dim(1).value();
  int32_t width = input_shape.dim(2).value();
  int32_t depth = input_shape.dim(3).value();

  int block_size = node->block_size();

  if (block_size < 2)
    INTERNAL_EXN("Block size must be >= 2");

  if ((height % block_size) || (width % block_size))
  {
    INTERNAL_EXN("The input tensor's height and width must be divisible by block_size");
  }

  loco::TensorShape output_shape;
  output_shape.rank(4);

  output_shape.dim(0) = input_shape.dim(0).value();
  output_shape.dim(1) = height / block_size;
  output_shape.dim(2) = width / block_size;
  output_shape.dim(3) = block_size * block_size * depth;

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_sparse_to_dense(const luci::CircleSparseToDense *node)
{
  loco::TensorShape shape;
  {
    LUCI_ASSERT(node->output_shape(), "dims input should not be nullptr");

    auto output_shape_node = dynamic_cast<luci::CircleConst *>(node->output_shape());
    if (output_shape_node != nullptr)
    {
      const auto output_shape_type = output_shape_node->dtype();

      if (output_shape_node->rank() != 1)
        INTERNAL_EXN_V("Only support rank 1 CircleConst",
                       oops::to_uint32(output_shape_node->rank()));

      if (output_shape_type == loco::DataType::S32)
      {
        shape.rank(output_shape_node->size<loco::DataType::S32>());

        for (uint32_t axis = 0; axis < shape.rank(); ++axis)
        {
          shape.dim(axis) = output_shape_node->at<loco::DataType::S32>(axis);
        }
      }
      else if (output_shape_type == loco::DataType::S64)
      {
        shape.rank(output_shape_node->size<loco::DataType::S64>());

        for (uint32_t axis = 0; axis < shape.rank(); ++axis)
        {
          shape.dim(axis) = output_shape_node->at<loco::DataType::S64>(axis);
        }
      }
      else
      {
        INTERNAL_EXN("Output shape of SparseToDense must be either int32 or int64");
      }
    }
    else
    {
      shape = own_shape(node);
    }
  }

  return loco::NodeShape{shape};
}

loco::NodeShape infer_strided_slice(const luci::CircleStridedSlice *node)
{
  auto begin_node = dynamic_cast<luci::CircleConst *>(node->begin());
  auto end_node = dynamic_cast<luci::CircleConst *>(node->end());
  auto strides_node = dynamic_cast<luci::CircleConst *>(node->strides());

  if (begin_node == nullptr || end_node == nullptr || strides_node == nullptr)
  {
    return use_own(node);
  }

  loco::TensorShape shape = infer_output_shape(node);
  return loco::NodeShape{shape};
}

loco::NodeShape infer_squeeze(const luci::CircleSqueeze *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

  // TODO input shape may be unknown before runtime
  std::vector<bool> do_squeeze(input_shape.rank(), false);
  uint32_t num_squeezed = 0;

  if (!node->squeeze_dims().empty())
  {
    // SqueezeDims not empty, squeeze only dims specified
    for (int32_t raw_dim : node->squeeze_dims())
    {
      int32_t dim = raw_dim < 0 ? raw_dim + input_shape.rank() : raw_dim;

      if (dim < 0 || static_cast<uint32_t>(dim) >= input_shape.rank() ||
          input_shape.dim(dim).value() != 1)
      {
        INTERNAL_EXN("invalid dimention specified to Squeeze");
      }

      if (!do_squeeze[dim])
        ++num_squeezed;
      do_squeeze[dim] = true;
    }
  }
  else
  {
    // SqueezeDims empty, squeeze any dims with size == 1
    for (uint32_t dim = 0; dim < input_shape.rank(); ++dim)
    {
      if (input_shape.dim(dim) == 1)
      {
        do_squeeze[dim] = true;
        ++num_squeezed;
      }
    }
  }

  loco::TensorShape output_shape;
  output_shape.rank(input_shape.rank() - num_squeezed);

  for (uint32_t in_dim = 0, out_dim = 0; in_dim < input_shape.rank(); ++in_dim)
  {
    if (!do_squeeze[in_dim])
    {
      output_shape.dim(out_dim++) = input_shape.dim(in_dim);
    }
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_svdf(const luci::CircleSVDF *node)
{
  const auto ifm_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  const auto weight_feature_shape = luci::shape_get(node->weight_feature()).as<loco::TensorShape>();

  assert(ifm_shape.rank() == 2);
  assert(weight_feature_shape.rank() == 2);

  assert(ifm_shape.dim(1) == weight_feature_shape.dim(1));
  assert(weight_feature_shape.dim(0).known());

  const auto rank = node->svdf_rank();
  const auto num_filters = weight_feature_shape.dim(0).value();
  assert(num_filters % rank == 0);
  const auto num_units = num_filters / rank;

  loco::TensorShape ofm_shape;
  ofm_shape.rank(2);
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(1) = num_units;

  return loco::NodeShape{ofm_shape};
}

loco::NodeShape infer_tile(const luci::CircleTile *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto multiples = loco::must_cast<luci::CircleConst *>(node->multiples());

  // TODO support non-const case
  // TODO support S64 type
  LUCI_ASSERT(multiples->dtype() == S32, "Only support int32 multiples");
  LUCI_ASSERT(multiples->rank() == 1, "multiples should be rank 1")

  uint32_t n = multiples->dim(0).value();

  LUCI_ASSERT(n == input_shape.rank(), "length of multiples should be the same with input rank");

  loco::TensorShape output_shape;

  output_shape.rank(input_shape.rank());
  for (uint32_t ni = 0; ni < n; ++ni)
  {
    int32_t multiple = multiples->at<S32>(ni);
    output_shape.dim(ni) = input_shape.dim(ni).value() * static_cast<uint32_t>(multiple);
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_transpose(const luci::CircleTranspose *node)
{
  auto input_shape = luci::shape_get(node->a()).as<loco::TensorShape>();

  auto perm_node = loco::must_cast<luci::CircleConst *>(node->perm());

  loco::TensorShape output_shape;
  output_shape.rank(input_shape.rank());

  assert(perm_node->dtype() == loco::DataType::S32);
  assert(input_shape.rank() == perm_node->template size<loco::DataType::S32>());

  for (uint32_t out_axis = 0; out_axis < output_shape.rank(); out_axis++)
  {
    auto in_axis = perm_node->template at<loco::DataType::S32>(out_axis);
    output_shape.dim(out_axis) = input_shape.dim(in_axis);
  }

  return output_shape;
}

loco::NodeShape infer_transpose_conv(const luci::CircleTransposeConv *node)
{
  // TransposeConv's output shape is written in its 'inputSizes' argument
  auto input_sizes_const = loco::must_cast<luci::CircleConst *>(node->inputSizes());
  // TODO support non-const type
  LUCI_ASSERT(input_sizes_const->dtype() == loco::DataType::S32, "Only support S32 dtype")
  LUCI_ASSERT(input_sizes_const->rank() == 1 && input_sizes_const->dim(0).value() == 4,
              "Only support rank 1 with 4 entries")

  loco::TensorShape shape;

  shape.rank(4);
  for (uint32_t axis = 0; axis < 4; ++axis)
    shape.dim(axis) = input_sizes_const->at<loco::DataType::S32>(axis);

  return loco::NodeShape{shape};
}

loco::NodeShape infer_unpack(const luci::CircleUnpack *node)
{
  // CircleUnpack provides list(array) of Tensors which has one less dimension of the input
  // We'll set shape of CircleUnpack to shape of actual outputs
  // TODO fix this if any problem rises
  auto value_shape = luci::shape_get(node->value()).as<loco::TensorShape>();

  auto axis = node->axis();
  auto num = node->num();
  auto rank = static_cast<int32_t>(value_shape.rank());

  if (rank == 0)
  {
    // Unknown shape
    return use_own(node);
  }

  LUCI_ASSERT(-rank <= axis && axis < rank, "Axis is out of range");

  if (axis < 0)
    axis += rank;

  LUCI_ASSERT(num == static_cast<int32_t>(value_shape.dim(axis).value()),
              "num, axis maybe incorrect");

  loco::TensorShape output_shape;
  output_shape.rank(rank - 1);

  for (int32_t i = 0, o = 0; i < rank; ++i)
  {
    if (i != axis)
      output_shape.dim(o++) = value_shape.dim(i);
  }

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_unidirectionalsequencelstm(const luci::CircleUnidirectionalSequenceLSTM *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto recurrent_to_output_weights =
    luci::shape_get(node->recurrent_to_output_weights()).as<loco::TensorShape>();
  auto rank = input_shape.rank();
  loco::TensorShape output_shape;
  output_shape.rank(rank);
  for (uint32_t i = 0; i < rank - 1; i++)
  {
    output_shape.dim(i) = input_shape.dim(i);
  }
  output_shape.dim(rank - 1) = recurrent_to_output_weights.dim(1);
  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_unique(const luci::CircleUnique *node)
{
  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

  assert(input_shape.rank() == 1);

  loco::TensorShape shape_output;
  shape_output = own_shape(node);

  return loco::NodeShape{shape_output};
}

// Circle Only
loco::NodeShape infer_bcq_fully_connected(const luci::CircleBCQFullyConnected *node)
{
  loco::TensorShape out_shape;

  auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
  auto weights_clusters = loco::must_cast<luci::CircleConst *>(node->weights_clusters());

  LUCI_ASSERT(input_shape.rank() == 2, "Input rank of BCQFullyConnected should be 2");

  int32_t qbits_sum = 0;
  for (uint32_t i = 0; i < weights_clusters->dim(0).value(); ++i)
  {
    qbits_sum += weights_clusters->at<loco::DataType::S32>(i * 2 + 1);
  }

  out_shape.rank(2);
  out_shape.dim(0) = qbits_sum;
  out_shape.dim(1) = input_shape.dim(1);

  return loco::NodeShape{out_shape};
}

loco::NodeShape infer_bcq_gather(const luci::CircleBCQGather *node)
{
  loco::TensorShape input_shape;
  loco::TensorShape output_shape;

  const auto input_binary_shape = luci::shape_get(node->input_binary()).as<loco::TensorShape>();
  const auto indices_shape = luci::shape_get(node->indices()).as<loco::TensorShape>();
  auto axis = node->axis();

  auto input_clusters = loco::must_cast<luci::CircleConst *>(node->input_clusters());
  auto qbits_sum = 0;
  for (uint32_t i = 0; i < input_clusters->dim(0).value(); ++i)
  {
    qbits_sum += input_clusters->at<loco::DataType::S32>(i * 2 + 1);
  }

  input_shape.rank(2);
  input_shape.dim(0) = qbits_sum;
  input_shape.dim(1) = input_binary_shape.dim(1).value() * 32;

  output_shape.rank(input_shape.rank() - 1 + indices_shape.rank());
  int32_t outdim_index = 0;
  for (int32_t i = 0; i < axis; ++i)
    output_shape.dim(outdim_index++) = input_shape.dim(i);
  for (uint32_t i = 0; i < indices_shape.rank(); ++i)
    output_shape.dim(outdim_index++) = indices_shape.dim(i);
  for (uint32_t i = axis + 1; i < input_shape.rank(); ++i)
    output_shape.dim(outdim_index++) = input_shape.dim(i);

  return loco::NodeShape{output_shape};
}

// Virtual
loco::NodeShape infer_input(const luci::CircleInput *node)
{
  loco::TensorShape shape;

  shape.rank(node->rank());
  for (uint32_t axis = 0; axis < node->rank(); axis++)
    shape.dim(axis) = node->dim(axis);

  return loco::NodeShape{shape};
}

loco::NodeShape infer_output(const luci::CircleOutput *node)
{
  auto graph_outputs = node->graph()->outputs();
  auto graph_output = graph_outputs->at(node->index());
  auto output_shape = graph_output->shape();

  return loco::NodeShape{*output_shape};
}

loco::NodeShape infer_non_max_suppression_v4_out(const luci::CircleNonMaxSuppressionV4Out *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto nmsv4 = dynamic_cast<const luci::CircleNonMaxSuppressionV4 *>(node->input());
  if (nmsv4 == nullptr)
    INTERNAL_EXN("CircleNonMaxSuppressionV4 IR is not configured correctly");

  auto index = node->index();
  if (index == 1)
    return loco::TensorShape({0});

  assert(index == 0);

  auto unknown = loco::TensorShape{loco::Dimension()};
  auto max_output_size = dynamic_cast<const luci::CircleConst *>(nmsv4->max_output_size());
  if (max_output_size == nullptr)
    return unknown; // we need CircleConst for max output size

  LUCI_ASSERT(max_output_size->dtype() == S32, "Only support int32 for max_output_size");

  if (max_output_size->size<S32>() < 1)
    return unknown;

  auto max_output_size_value = uint32_t(max_output_size->at<S32>(0));
  return loco::TensorShape{max_output_size_value};
}

loco::NodeShape infer_non_max_suppression_v5_out(const luci::CircleNonMaxSuppressionV5Out *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto nmsv5 = dynamic_cast<const luci::CircleNonMaxSuppressionV5 *>(node->input());
  if (nmsv5 == nullptr)
    INTERNAL_EXN("CircleNonMaxSuppressionV5 IR is not configured correctly");

  auto index = node->index();
  if (index == 2)
    return loco::TensorShape({0});

  assert(index == 0 || index == 1);

  auto unknown = loco::TensorShape{loco::Dimension()};
  auto max_output_size = dynamic_cast<const luci::CircleConst *>(nmsv5->max_output_size());
  if (max_output_size == nullptr)
    return unknown; // we need CircleConst for max output size

  LUCI_ASSERT(max_output_size->dtype() == S32, "Only support int32 for max_output_size");

  if (max_output_size->size<S32>() < 1)
    return unknown;

  auto max_output_size_value = uint32_t(max_output_size->at<S32>(0));
  return loco::TensorShape{max_output_size_value};
}

loco::NodeShape infer_split_out(const luci::CircleSplitOut *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto split = dynamic_cast<const luci::CircleSplit *>(node->input());
  if (split == nullptr)
    INTERNAL_EXN("CircleSplit IR is not configured correctly");

  loco::NodeShape unknown;

  auto split_shape = luci::shape_get(split).as<loco::TensorShape>();

  auto split_dim = dynamic_cast<const luci::CircleConst *>(split->split_dim());
  if (split_dim == nullptr)
    return unknown; // we need CircleConst for split_dim
  LUCI_ASSERT(split_dim->dtype() == S32, "Only support int32 for split_dim");

  assert(split_dim->size<S32>() == 1);
  auto split_dim_axis = split_dim->at<S32>(0);
  if (split_dim_axis < 0)
    split_dim_axis += split_shape.rank();

  auto split_dim_value = split_shape.dim(split_dim_axis).value();
  assert(split_dim_value % split->num_split() == 0);
  const int split_depth = split_dim_value / split->num_split();

  loco::TensorShape output_shape = split_shape;

  // All shapes are equally same
  output_shape.dim(split_dim_axis) = loco::Dimension(split_depth);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_split_v_out(const luci::CircleSplitVOut *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto split = dynamic_cast<const luci::CircleSplitV *>(node->input());
  if (split == nullptr)
    INTERNAL_EXN("CircleSplit IR is not configured correctly");

  loco::NodeShape unknown;

  auto split_shape = luci::shape_get(split).as<loco::TensorShape>();

  auto size_splits = dynamic_cast<const luci::CircleConst *>(split->size_splits());
  if (size_splits == nullptr)
    return unknown; // we need CircleConst for size_splits
  LUCI_ASSERT(size_splits->dtype() == S32, "Only support int32 for size_splits");

  auto split_dim = dynamic_cast<const luci::CircleConst *>(split->split_dim());
  if (split_dim == nullptr)
    return unknown; // we need CircleConst for split_dim
  LUCI_ASSERT(split_dim->dtype() == S32, "Only support int32 for split_dim");

  // fetch axis
  assert(split_dim->size<S32>() == 1);
  auto split_dim_axis = split_dim->at<S32>(0);
  if (split_dim_axis < 0)
    split_dim_axis += split_shape.rank();

  // interpret size_splits values
  int32_t size_splits_count = static_cast<int32_t>(size_splits->size<S32>());
  assert(size_splits_count == split->num_split());

  int64_t minus_one_count = 0, size_splits_sum = 0;
  for (int32_t idx = 0; idx < size_splits_count; ++idx)
  {
    auto size = size_splits->at<S32>(idx);
    assert(size >= -1);
    if (size == -1)
      ++minus_one_count;
    else
      size_splits_sum += size;
  }
  if (minus_one_count > 1)
    INTERNAL_EXN("CircleSplitV size_splits has more than two -1 values");

  // calcuate this SplitVOut shape
  auto input_size = split_shape.dim(split_dim_axis).value();
  assert(size_splits_sum <= input_size);

  auto index_this = node->index();
  assert(0 <= index_this && index_this < split->num_split());
  auto split_depth = size_splits->at<S32>(index_this);
  if (split_depth == -1)
    split_depth = static_cast<int32_t>(input_size) - static_cast<int32_t>(size_splits_sum);

  loco::TensorShape output_shape = split_shape;

  output_shape.dim(split_dim_axis) = loco::Dimension(split_depth);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_top_k_v2_out(const luci::CircleTopKV2Out *node)
{
  const loco::DataType S32 = loco::DataType::S32;

  auto topkv2 = dynamic_cast<const luci::CircleTopKV2 *>(node->input());
  if (topkv2 == nullptr)
    INTERNAL_EXN("CircleSplit IR is not configured correctly");

  // shape of topkv2 is same as topkv2->input()
  auto input_shape = luci::shape_get(topkv2).as<loco::TensorShape>();

  auto node_k = loco::must_cast<const luci::CircleConst *>(topkv2->k());
  LUCI_ASSERT(node_k->dtype() == S32, "Only support Int32");
  assert(node_k->size<S32>() == 1);

  loco::TensorShape output_shape;

  output_shape.rank(input_shape.rank());
  for (uint32_t idx = 0; idx < input_shape.rank() - 1; ++idx)
  {
    output_shape.dim(idx) = input_shape.dim(idx);
  }
  output_shape.dim(input_shape.rank() - 1) = node_k->at<S32>(0);

  return loco::NodeShape{output_shape};
}

loco::NodeShape infer_unique_out(const luci::CircleUniqueOut *node)
{
  if (node->index() == 0)
  {
    auto unique_shape = own_shape(node);
    return loco::NodeShape{unique_shape};
  }
  assert(node->index() == 1);
  auto unique = loco::must_cast<luci::CircleUnique *>(node->input());
  auto unique_shape = luci::shape_get(unique->input()).as<loco::TensorShape>();

  assert(unique_shape.rank() == 1);

  loco::TensorShape shape_output;
  shape_output.rank(1);
  shape_output.dim(0) = unique_shape.dim(0);
  return loco::NodeShape{shape_output};
}

loco::NodeShape infer_unpack_out(const luci::CircleUnpackOut *node)
{
  auto unpack = dynamic_cast<const luci::CircleUnpack *>(node->input());
  if (unpack == nullptr)
  {
    INTERNAL_EXN("CircleUnpack IR is not configured correctly");
  }

  auto unpack_shape = luci::shape_get(unpack).as<loco::TensorShape>();

  return loco::NodeShape{unpack_shape};
}

loco::NodeShape infer_while_out(const luci::CircleWhileOut *node)
{
  /**
   * @note  WHILE operator's shape is the same with the "cond"
   *        Graph input.
   */
  auto circle_while = dynamic_cast<const luci::CircleWhile *>(node->input());
  if (circle_while == nullptr)
  {
    INTERNAL_EXN("CircleWhile IR is not configured correctly");
  }

  auto index = node->index();
  auto cond_graph = circle_while->cond_graph();
  assert(cond_graph != nullptr);

  // Assumption: the index of CircleWhileOut matches with the index of input nodes returned by
  // loco::input_nodes
  auto cond_inputs = loco::input_nodes(cond_graph);
  auto cond_in = loco::must_cast<luci::CircleInput *>(cond_inputs.at(index));

  auto cond_graph_inputs = cond_graph->inputs();
  auto cond_graph_input = cond_graph_inputs->at(cond_in->index());

  auto cond_graph_input_shape = *cond_graph_input->shape();
  auto this_shape = own_shape(node);

  if (!(this_shape == cond_graph_input_shape))
  {
    LOGGER(l);
    WARN(l) << "Warning: CircleWhileOut '" << node->name() << "' shape mispatch " << this_shape
            << " vs " << cond_graph_input_shape;
  }

  return loco::NodeShape{this_shape};
}

/**
 * @brief Class to infer the shape of CircleNode
 *
 * @note All CircleNode's inputs and outputs are always loco::Domain::Tensor
 */
class ShapeInferenceAlgorithm final : public luci::CircleNodeVisitor<loco::NodeShape>
{
public:
  loco::NodeShape visit(const luci::CircleAbs *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleAdd *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleAddN *node) final { return infer_add_n(node); }

  loco::NodeShape visit(const luci::CircleArgMax *node) final { return infer_arg_maxmin(node); }

  loco::NodeShape visit(const luci::CircleArgMin *node) final { return infer_arg_maxmin(node); }

  loco::NodeShape visit(const luci::CircleAveragePool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const luci::CircleBatchMatMul *node) final
  {
    auto x_shape = luci::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = luci::shape_get(node->y()).as<loco::TensorShape>();

    return infer_batchmatmul_shape(x_shape, y_shape, node->adj_x(), node->adj_y());
  }

  loco::NodeShape visit(const luci::CircleBatchToSpaceND *node) final
  {
    return infer_batch_to_space_nd(node);
  }

  loco::NodeShape visit(const luci::CircleBroadcastTo *node) final
  {
    return infer_broadcast_to(node);
  }

  loco::NodeShape visit(const luci::CircleCast *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleCeil *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleConcatenation *node) final
  {
    return infer_concatenation(node);
  }

  loco::NodeShape visit(const luci::CircleConst *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleConv2D *node) final { return infer_conv2d(node); }

  loco::NodeShape visit(const luci::CircleCos *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleCumSum *node) final { return use_input(node); }

  loco::NodeShape visit(const luci::CircleCustom *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleDensify *node) final { return use_input(node); }

  loco::NodeShape visit(const luci::CircleDepthToSpace *node) final
  {
    return infer_depth_to_space(node);
  }

  loco::NodeShape visit(const luci::CircleDepthwiseConv2D *node) final
  {
    return infer_depthwise_conv2d(node);
  }

  loco::NodeShape visit(const luci::CircleDequantize *node) final
  {
    const auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleDiv *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleElu *node) final
  {
    auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleEqual *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleExp *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleExpandDims *node) final
  {
    return infer_expand_dims(node);
  }

  loco::NodeShape visit(const luci::CircleFakeQuant *node) final { return use_inputs(node); }

  loco::NodeShape visit(const luci::CircleFill *node) final { return infer_fill(node); }

  loco::NodeShape visit(const luci::CircleFloor *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleFloorDiv *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleFloorMod *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleFullyConnected *node) final
  {
    return infer_fully_connected(node);
  }

  loco::NodeShape visit(const luci::CircleGather *node) final { return infer_gather(node); }

  loco::NodeShape visit(const luci::CircleGatherNd *node) final { return infer_gather_nd(node); }

  loco::NodeShape visit(const luci::CircleGelu *node) final
  {
    auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleGreater *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleGreaterEqual *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleHardSwish *node) final
  {
    auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleIf *node) final
  {
    // Shape of CircleIf is not used. Just use input 0
    assert(node->input_count() > 0);
    const auto input_shape = luci::shape_get(node->input(0)).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleL2Normalize *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleL2Pool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const luci::CircleLeakyRelu *node) final
  {
    const auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleLess *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleLessEqual *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleLocalResponseNormalization *node) final
  {
    const auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleLog *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleLogicalAnd *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleLogicalNot *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleLogicalOr *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleLogistic *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleLogSoftmax *node) final { return use_logits(node); }

  loco::NodeShape visit(const luci::CircleMatrixDiag *node) final
  {
    return infer_matrix_diag(node);
  }

  loco::NodeShape visit(const luci::CircleMatrixSetDiag *node) final
  {
    return infer_matrix_set_diag(node);
  }

  loco::NodeShape visit(const luci::CircleMaximum *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleMaxPool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const luci::CircleMean *node) final
  {
    auto output_shape = infer_reducer(node->input(), node->reduction_indices(), node->keep_dims());
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleMinimum *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleMirrorPad *node) final { return infer_mirror_pad(node); }

  loco::NodeShape visit(const luci::CircleMul *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleNeg *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleNonMaxSuppressionV4 *node) final
  {
    const auto boxes_shape = luci::shape_get(node->boxes()).as<loco::TensorShape>();
    return loco::NodeShape{boxes_shape};
  }

  loco::NodeShape visit(const luci::CircleNonMaxSuppressionV5 *node) final
  {
    const auto boxes_shape = luci::shape_get(node->boxes()).as<loco::TensorShape>();
    return loco::NodeShape{boxes_shape};
  }

  loco::NodeShape visit(const luci::CircleNotEqual *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleOneHot *node) final { return infer_one_hot(node); }

  loco::NodeShape visit(const luci::CirclePack *node) final { return infer_pack(node); }

  loco::NodeShape visit(const luci::CirclePad *node) final { return infer_pad(node); }

  loco::NodeShape visit(const luci::CirclePadV2 *node) final { return infer_pad_v2(node); }

  loco::NodeShape visit(const luci::CirclePow *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CirclePRelu *node) final { return infer_p_relu(node); }

  loco::NodeShape visit(const luci::CircleQuantize *node) final
  {
    const auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleRange *node) final { return infer_range(node); }

  loco::NodeShape visit(const luci::CircleRank *) final
  {
    loco::TensorShape shape_output;
    shape_output.rank(0);

    return loco::NodeShape{shape_output};
  }

  loco::NodeShape visit(const luci::CircleReduceAny *node) final
  {
    auto output_shape = infer_reducer(node->input(), node->reduction_indices(), node->keep_dims());
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleReduceMax *node) final
  {
    auto output_shape = infer_reducer(node->input(), node->reduction_indices(), node->keep_dims());
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleReduceMin *node) final
  {
    auto output_shape = infer_reducer(node->input(), node->reduction_indices(), node->keep_dims());
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleReduceProd *node) final
  {
    auto output_shape = infer_reducer(node->input(), node->reduction_indices(), node->keep_dims());
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleRelu *node) final
  {
    auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleRelu6 *node) final
  {
    auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleReluN1To1 *node) final
  {
    auto input_shape = luci::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  /**
   * @note  CircleReshape has new shape info in two places: 2nd input and attribute.
   *        This shape inference uses shape from input 'shape' node when it's constant.
   *        If not, shape will be from node itself. shape from attribute is not used.
   *
   * TODO Change this policy when not appropriate
   */
  loco::NodeShape visit(const luci::CircleReshape *node) final { return infer_reshape(node); }

  loco::NodeShape visit(const luci::CircleResizeBilinear *node) final
  {
    return infer_resize_type(node);
  }

  loco::NodeShape visit(const luci::CircleResizeNearestNeighbor *node) final
  {
    return infer_resize_type(node);
  }

  loco::NodeShape visit(const luci::CircleReverseSequence *node) final
  {
    auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleRound *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleReverseV2 *node) final
  {
    auto input_shape = luci::shape_get(node->tensor()).as<loco::TensorShape>();

    LUCI_ASSERT(luci::shape_get(node->axis()).as<loco::TensorShape>().rank() == 1,
                "Tensor must be 1-D");

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleRsqrt *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleScatterNd *node) final { return infer_scatter_nd(node); }

  loco::NodeShape visit(const luci::CircleSegmentSum *node) final
  {
    return infer_segment_sum(node);
  }

  loco::NodeShape visit(const luci::CircleSelect *node) final { return infer_select(node); }

  loco::NodeShape visit(const luci::CircleSelectV2 *node) final { return infer_select_v2(node); }

  loco::NodeShape visit(const luci::CircleShape *node) final { return infer_shape(node); }

  loco::NodeShape visit(const luci::CircleSin *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleSlice *node) final { return infer_slice(node); }

  loco::NodeShape visit(const luci::CircleSoftmax *node) final { return use_logits(node); }

  loco::NodeShape visit(const luci::CircleSpaceToBatchND *node) final
  {
    return infer_space_to_batch_nd(node);
  }

  loco::NodeShape visit(const luci::CircleSpaceToDepth *node) final
  {
    return infer_space_to_depth(node);
  }

  loco::NodeShape visit(const luci::CircleSparseToDense *node) final
  {
    return infer_sparse_to_dense(node);
  }

  loco::NodeShape visit(const luci::CircleSplit *node) final
  {
    // We'll set Split output as same as input so that SplitOut can handle it's own shape
    auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleSplitV *node) final
  {
    // We'll set SplitV output as same as input so that SplitOut can handle it's own shape
    auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleSqrt *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleSquare *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleSquaredDifference *node) final
  {
    return broadcast_xy(node);
  }

  loco::NodeShape visit(const luci::CircleStridedSlice *node) final
  {
    return infer_strided_slice(node);
  }

  loco::NodeShape visit(const luci::CircleSqueeze *node) final { return infer_squeeze(node); }

  loco::NodeShape visit(const luci::CircleSub *node) final { return broadcast_xy(node); }

  loco::NodeShape visit(const luci::CircleSum *node) final
  {
    auto output_shape = infer_reducer(node->input(), node->reduction_indices(), node->keep_dims());
    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const luci::CircleSVDF *node) final { return infer_svdf(node); }

  loco::NodeShape visit(const luci::CircleTanh *node) final { return use_x(node); }

  loco::NodeShape visit(const luci::CircleTile *node) final { return infer_tile(node); }

  loco::NodeShape visit(const luci::CircleTopKV2 *node) final
  {
    // set shape of this node as same as input
    const auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleTranspose *node) final { return infer_transpose(node); }

  loco::NodeShape visit(const luci::CircleTransposeConv *node) final
  {
    return infer_transpose_conv(node);
  }

  loco::NodeShape visit(const luci::CircleUnpack *node) final { return infer_unpack(node); }

  loco::NodeShape visit(const luci::CircleUnidirectionalSequenceLSTM *node) final
  {
    return infer_unidirectionalsequencelstm(node);
  }

  loco::NodeShape visit(const luci::CircleUnique *node) final { return infer_unique(node); }

  loco::NodeShape visit(const luci::CircleWhere *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleWhile *node) final
  {
    // Shape of CircleWhile is not used. Just use input 0
    assert(node->arity() > 0);
    const auto input_shape = luci::shape_get(node->input(0)).as<loco::TensorShape>();
    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const luci::CircleZerosLike *node) final
  {
    auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  // Circle Only
  loco::NodeShape visit(const luci::CircleBCQFullyConnected *node) final
  {
    return infer_bcq_fully_connected(node);
  }

  loco::NodeShape visit(const luci::CircleBCQGather *node) final { return infer_bcq_gather(node); }

  loco::NodeShape visit(const luci::CircleInstanceNorm *node) final
  {
    auto input_shape = luci::shape_get(node->input()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  // Virtual
  loco::NodeShape visit(const luci::CircleInput *node) final { return infer_input(node); }

  loco::NodeShape visit(const luci::CircleOutput *node) final { return infer_output(node); }

  loco::NodeShape visit(const luci::CircleOutputDummy *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleOutputExclude *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleCustomOut *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleNonMaxSuppressionV4Out *node) final
  {
    return infer_non_max_suppression_v4_out(node);
  }

  loco::NodeShape visit(const luci::CircleNonMaxSuppressionV5Out *node) final
  {
    return infer_non_max_suppression_v5_out(node);
  }

  loco::NodeShape visit(const luci::CircleSplitOut *node) final { return infer_split_out(node); }

  loco::NodeShape visit(const luci::CircleSplitVOut *node) final { return infer_split_v_out(node); }

  loco::NodeShape visit(const luci::CircleTopKV2Out *node) final
  {
    return infer_top_k_v2_out(node);
  }

  loco::NodeShape visit(const luci::CircleUniqueOut *node) final { return infer_unique_out(node); }

  loco::NodeShape visit(const luci::CircleUnpackOut *node) final { return infer_unpack_out(node); }

  loco::NodeShape visit(const luci::CircleVariable *node) final { return use_own(node); }

  loco::NodeShape visit(const luci::CircleWhileOut *node) final { return infer_while_out(node); }
};

} // namespace

namespace luci
{

bool CircleShapeInferenceRule::recognize(const loco::Dialect *d) const
{
  return CircleDialect::get() == d;
}

bool CircleShapeInferenceRule::infer(const loco::Node *node, loco::NodeShape &shape) const
{
  LOGGER(l);

  assert(node->dialect() == CircleDialect::get());

  ShapeInferenceAlgorithm alg;
  auto circle_node = loco::must_cast<const CircleNode *>(node);

  bool is_shape_undefined = (circle_node->shape_status() == ShapeStatus::UNDEFINED);
  bool is_shape_none = (circle_node->shape_status() == ShapeStatus::NOSHAPE);
  bool is_scalar = (circle_node->rank() == 0);

  if (is_shape_undefined)
    shape = circle_node->accept(&alg);
  else
  {
    if (is_shape_none || is_scalar)
      shape = own_shape(circle_node);
    else
      shape = circle_node->accept(&alg);
  }

  VERBOSE(l, 1) << "[luci] shape: " << circle_node->name();
  VERBOSE(l, 1) << "              own_shape: " << own_shape(circle_node)
                << " -> infer: " << shape.as<loco::TensorShape>();

  return true;
}

} // namespace luci
