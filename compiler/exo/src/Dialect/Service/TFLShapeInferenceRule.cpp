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

#include "TFLShapeInferenceRule.h"

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLDialect.h"
#include "Dialect/IR/TFLNodeVisitor.h"

#include "Check.h"

#include <oops/InternalExn.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace
{

// Call this for TFLAvgPool2D and TFLMaxPool2D only
template <class Pool2DType> loco::NodeShape infer_pool_2d_shape(const Pool2DType *node)
{
  EXO_ASSERT(loco::shape_known(node->value()), "Shape must be known");

  auto ifm_shape = loco::shape_get(node->value()).template as<loco::TensorShape>();
  assert(ifm_shape.rank() == 4);

  uint32_t input_height = ifm_shape.dim(1).value();
  uint32_t input_width = ifm_shape.dim(2).value();
  uint32_t stride_height = node->stride()->h();
  uint32_t stride_width = node->stride()->w();
  uint32_t window_height = node->filter()->h();
  uint32_t window_width = node->filter()->w();
  uint32_t dilation_height = 1; // dilation for TFLAvgPool2D and TFLMaxPool2D is 1
  uint32_t dilation_width = 1;
  uint32_t effective_window_height = dilation_height * (window_height - 1) + 1;
  uint32_t effective_window_width = dilation_width * (window_width - 1) + 1;

  uint32_t output_height = 0;
  uint32_t output_width = 0;

  if (node->padding() == locoex::Padding::VALID)
  {
    output_height = (input_height + stride_height - effective_window_height) / stride_height;
    output_width = (input_width + stride_width - effective_window_width) / stride_width;
  }
  else if (node->padding() == locoex::Padding::SAME)
  {
    output_height = (input_height + stride_height - 1) / stride_height;
    output_width = (input_width + stride_width - 1) / stride_width;
  }
  else
    EXO_ASSERT(false, "Wrong padding type");

  loco::TensorShape ofm_shape;
  ofm_shape.rank(4);
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(1) = output_height;
  ofm_shape.dim(2) = output_width;
  ofm_shape.dim(3) = ifm_shape.dim(3);

  return loco::NodeShape{ofm_shape};
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
    assert(x.dim(axis).known() && y.dim(axis).known());

    auto x_dim = x.dim(axis).value();
    auto y_dim = y.dim(axis).value();

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
 * @brief Class to infer the shape of TFLNode
 *
 * @note All TFLNode's inputs and outputs are always loco::Domain::Tensor
 */
class ShapeInferenceAlgorithm final : public locoex::TFLNodeVisitor<loco::NodeShape>
{
public:
  loco::NodeShape visit(const locoex::TFLAdd *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLAveragePool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const locoex::TFLConcatenation *node) final
  {
    // TODO Support when TFLConcatenation has 0 input
    assert(node->numValues() > 0);

    auto axis = node->axis();
    auto first_shape = loco::shape_get(node->values(0)).as<loco::TensorShape>();

    loco::TensorShape output_shape;

    output_shape.rank(first_shape.rank());
    for (uint32_t i = 0; i < output_shape.rank(); ++i)
      output_shape.dim(i) = first_shape.dim(i);

    for (uint32_t i = 1; i < node->numValues(); ++i)
    {
      auto input_shape = loco::shape_get(node->values(i)).as<loco::TensorShape>();

      for (uint32_t j = 0; j < output_shape.rank(); ++j)
      {
        if (j == axis)
          output_shape.dim(j) = output_shape.dim(j).value() + input_shape.dim(j).value();
        else
          assert(output_shape.dim(j) == input_shape.dim(j));
      }
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLConst *node) final
  {
    loco::TensorShape shape;

    shape.rank(node->rank());
    for (uint32_t axis = 0; axis < node->rank(); axis++)
      shape.dim(axis) = node->dim(axis);

    return loco::NodeShape{shape};
  }

  loco::NodeShape visit(const locoex::TFLConv2D *node) final
  {
    auto ifm_shape = loco::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
    auto ker_shape = loco::shape_get(node->filter()).as<loco::TensorShape>(); // in OHWI

    assert(ifm_shape.rank() == 4);
    assert(ker_shape.rank() == 4);
    assert(ifm_shape.dim(3) == ker_shape.dim(3));

    uint32_t input_height = ifm_shape.dim(1).value();
    uint32_t input_width = ifm_shape.dim(2).value();
    uint32_t stride_height = node->stride()->h();
    uint32_t stride_width = node->stride()->w();
    uint32_t ker_height = ker_shape.dim(1).value();
    uint32_t ker_width = ker_shape.dim(2).value();
    uint32_t dilation_height = 1;
    uint32_t dilation_width = 1;
    uint32_t effective_ker_height = dilation_height * (ker_height - 1) + 1;
    uint32_t effective_ker_width = dilation_width * (ker_width - 1) + 1;

    uint32_t output_height = 0;
    uint32_t output_width = 0;

    if (node->padding() == locoex::Padding::VALID)
    {
      output_height = (input_height + stride_height - effective_ker_height) / stride_height;
      output_width = (input_width + stride_width - effective_ker_width) / stride_width;
    }
    else if (node->padding() == locoex::Padding::SAME)
    {
      output_height = (input_height + stride_height - 1) / stride_height;
      output_width = (input_width + stride_width - 1) / stride_width;
    }
    else
      EXO_ASSERT(false, "Wrong padding type");

    loco::TensorShape ofm_shape;
    ofm_shape.rank(4);
    ofm_shape.dim(0) = ifm_shape.dim(0);
    ofm_shape.dim(1) = output_height;
    ofm_shape.dim(2) = output_width;
    ofm_shape.dim(3) = ker_shape.dim(0);

    return loco::NodeShape{ofm_shape};
  }

  loco::NodeShape visit(const locoex::TFLDepthwiseConv2D *node) final
  {
    auto ifm_shape = loco::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
    auto ker_shape = loco::shape_get(node->filter()).as<loco::TensorShape>(); // in 1 H W CM

    assert(ifm_shape.rank() == 4);
    assert(ker_shape.rank() == 4);
    assert(ker_shape.dim(0).value() == 1);

    uint32_t input_height = ifm_shape.dim(1).value();
    uint32_t input_width = ifm_shape.dim(2).value();
    uint32_t stride_height = node->stride()->h();
    uint32_t stride_width = node->stride()->w();
    uint32_t ker_height = ker_shape.dim(1).value();
    uint32_t ker_width = ker_shape.dim(2).value();
    uint32_t dilation_height = 1;
    uint32_t dilation_width = 1;
    uint32_t effective_ker_height = dilation_height * (ker_height - 1) + 1;
    uint32_t effective_ker_width = dilation_width * (ker_width - 1) + 1;

    uint32_t output_height = 0;
    uint32_t output_width = 0;

    if (node->padding() == locoex::Padding::VALID)
    {
      output_height = (input_height + stride_height - effective_ker_height) / stride_height;
      output_width = (input_width + stride_width - effective_ker_width) / stride_width;
    }
    else if (node->padding() == locoex::Padding::SAME)
    {
      output_height = (input_height + stride_height - 1) / stride_height;
      output_width = (input_width + stride_width - 1) / stride_width;
    }
    else
      EXO_ASSERT(false, "Wrong padding type");

    loco::TensorShape ofm_shape;
    ofm_shape.rank(4);
    ofm_shape.dim(0) = ifm_shape.dim(0);
    ofm_shape.dim(1) = output_height;
    ofm_shape.dim(2) = output_width;
    ofm_shape.dim(3) = ker_shape.dim(3);

    return loco::NodeShape{ofm_shape};
  }

  loco::NodeShape visit(const locoex::TFLDiv *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLFullyConnected *node) final
  {
    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto weights_shape = loco::shape_get(node->weights()).as<loco::TensorShape>();

    // Checking shape capability for multiplication
    EXO_ASSERT(input_shape.rank() == 2, "NYI for input shape rank > 2");
    EXO_ASSERT(weights_shape.rank() == 2, "Incompatible weights rank for fully connected");
    EXO_ASSERT(input_shape.dim(1) == weights_shape.dim(1),
               "Incompatible shapes for fully connected");

    loco::TensorShape out_shape;
    out_shape.rank(2);

    out_shape.dim(0) = input_shape.dim(0);
    out_shape.dim(1) = weights_shape.dim(0);

    return loco::NodeShape{out_shape};
  }

  loco::NodeShape visit(const locoex::TFLMaximum *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLMaxPool2D *node) final
  {
    return infer_pool_2d_shape(node);
  }

  loco::NodeShape visit(const locoex::TFLMean *node) final
  {
    const loco::DataType S32 = loco::DataType::S32;

    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto reduction_indices = dynamic_cast<locoex::TFLConst *>(node->reduction_indices());

    { // Exceptions
      // TODO support non-const case
      EXO_ASSERT(reduction_indices, "Only support constant reduction_indices");
      // TODO support other data type
      EXO_ASSERT(reduction_indices->dtype() == S32, "Only support int 32");
    }

    std::vector<int32_t> reduction_values;

    for (uint32_t i = 0; i < reduction_indices->size<S32>(); ++i)
    {
      int32_t axis = reduction_indices->at<S32>(i);
      if (axis < 0)
        axis += input_shape.rank();
      if (not(0 <= axis and axis < static_cast<int32_t>(input_shape.rank())))
        INTERNAL_EXN_V("Invalid reduction axis for MEAN", oops::to_uint32(axis));
      reduction_values.push_back(axis);
    }

    loco::TensorShape output_shape;

    if (node->keep_dims())
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
          output_shape.dim(j++) = i;
    }

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLMul *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLRelu *node) final
  {
    auto input_shape = loco::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const locoex::TFLRelu6 *node) final
  {
    auto input_shape = loco::shape_get(node->features()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  /**
   * @note  TFLReshape has new shape info in two places: 2nd input and attribute.
   *        This shape inference forces both to exist, and match each other.
   *        When this condition satisfied, it return the inferred shape
   *
   * TODO Change this policy when not appropriate
   */
  loco::NodeShape visit(const locoex::TFLReshape *node) final
  {
    const loco::DataType S32 = loco::DataType::S32;

    loco::TensorShape shape_by_input;
    {
      EXO_ASSERT(node->shape(), "2nd input shape() should not be nullptr");

      // Only support node's shape() is TFLConst with S32
      // TODO support other node with other types
      auto const_shape_node = dynamic_cast<locoex::TFLConst *>(node->shape());
      EXO_ASSERT(const_shape_node, "Only support TFLConst for shape of TFLReshape");
      EXO_ASSERT(const_shape_node->dtype() == S32, "Only support int32 TFLConst");

      if (const_shape_node->rank() != 1)
        INTERNAL_EXN_V("Only support rank 1 TFLConst", oops::to_uint32(const_shape_node->rank()));

      shape_by_input.rank(const_shape_node->dim(0).value());

      for (uint32_t axis = 0; axis < shape_by_input.rank(); ++axis)
      {
        EXO_ASSERT(const_shape_node->at<S32>(axis) > 0, "Dimension should be > 0")
        shape_by_input.dim(axis) = const_shape_node->at<S32>(axis);
      }
    }

    loco::TensorShape shape_by_attr;
    {
      shape_by_attr.rank(node->newShape()->rank());

      for (uint32_t axis = 0; axis < shape_by_attr.rank(); ++axis)
      {
        EXO_ASSERT(node->newShape()->dim(axis) > 0, "Dimension should be > 0")
        shape_by_attr.dim(axis) = node->newShape()->dim(axis);
      }
    }

    EXO_ASSERT(shape_by_input == shape_by_attr,
               "Warning: Two new shape information mismatched for TFLReshape");

    return loco::NodeShape{shape_by_input};
  }

  loco::NodeShape visit(const locoex::TFLRsqrt *node) final
  {
    auto input_shape = loco::shape_get(node->x()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  // TODO TFLSoftmax

  loco::NodeShape visit(const locoex::TFLSqrt *node) final
  {
    auto input_shape = loco::shape_get(node->x()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }

  loco::NodeShape visit(const locoex::TFLSquaredDifference *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  loco::NodeShape visit(const locoex::TFLSub *node) final
  {
    auto x_shape = loco::shape_get(node->x()).as<loco::TensorShape>();
    auto y_shape = loco::shape_get(node->y()).as<loco::TensorShape>();

    auto output_shape = broadcast_shape(x_shape, y_shape);

    return loco::NodeShape{output_shape};
  }

  // TODO TFLTanh

  /// @brief Returns output shape of transpose. Use loco::ConstGen and locoex::TFLConst for ConstT.
  template <class ConstT>
  loco::TensorShape output_shape_of_transpose(loco::TensorShape input_shape,
                                              const ConstT *perm_node)
  {
    loco::TensorShape output_shape;
    output_shape.rank(input_shape.rank());

    assert(perm_node->dtype() == loco::DataType::S32);
    assert(input_shape.rank() == perm_node->template size<loco::DataType::S32>());

    for (uint32_t out_axis = 0; out_axis < output_shape.rank(); out_axis++)
    {
      auto new_dim = perm_node->template at<loco::DataType::S32>(out_axis);
      output_shape.dim(new_dim) = input_shape.dim(out_axis);
    }

    return output_shape;
  }

  loco::NodeShape visit(const locoex::TFLTranspose *node) final
  {
    auto input_shape = loco::shape_get(node->a()).as<loco::TensorShape>();

    auto canon_perm = dynamic_cast<loco::ConstGen *>(node->perm());
    auto tfl_perm = dynamic_cast<locoex::TFLConst *>(node->perm());

    if (canon_perm)
    {
      return loco::NodeShape{output_shape_of_transpose(input_shape, canon_perm)};
    }
    else if (tfl_perm)
    {
      return loco::NodeShape{output_shape_of_transpose(input_shape, tfl_perm)};
    }
    else
      INTERNAL_EXN("perm of TFLTranspose should be either ConstGen or TFLConst");
  }

  loco::NodeShape visit(const locoex::TFLTransposeConv *node) final
  {
    // TransposeConv's output shape is written in its 'inputSizes' argument
    auto input_sizes_const = dynamic_cast<locoex::TFLConst *>(node->inputSizes());
    EXO_ASSERT(input_sizes_const, "Only support when TFLTransposeConv's inputSizes is TFLConst")
    EXO_ASSERT(input_sizes_const->dtype() == loco::DataType::S32, "Only support S32 dtype")
    EXO_ASSERT(input_sizes_const->rank() == 1 && input_sizes_const->dim(0).value() == 4,
               "Only support rank 1 with 4 entries")

    loco::TensorShape shape;

    shape.rank(4);
    for (uint32_t axis = 0; axis < 4; ++axis)
      shape.dim(axis) = input_sizes_const->at<loco::DataType::S32>(axis);

    return loco::NodeShape{shape};
  }
};

} // namespace

namespace locoex
{

bool TFLShapeInferenceRule::recognize(const loco::Dialect *d) const
{
  return TFLDialect::get() == d;
}

bool TFLShapeInferenceRule::infer(const loco::Node *node, loco::NodeShape &shape) const
{
  assert(node->dialect() == TFLDialect::get());
  assert(dynamic_cast<const TFLNode *>(node) != nullptr);

  ShapeInferenceAlgorithm alg;
  shape = dynamic_cast<const TFLNode *>(node)->accept(&alg);

  return true;
}

} // namespace locoex
