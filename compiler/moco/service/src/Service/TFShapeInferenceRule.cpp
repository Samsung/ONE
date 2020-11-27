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

#include "moco/Service/TFShapeInferenceRule.h"

#include <moco/Support/TFShapeInferenceHelper.h>

#include "moco/IR/TFDialect.h"
#include "moco/IR/TFNode.h"

#include <loco/IR/NodeShape.h>
#include <loco/Service/ShapeInference.h>

#include <oops/UserExn.h>

#include <cassert>
#include <cmath>

namespace
{

class ShapeInferenceAlgorithm final : public moco::TFNodeVisitor<loco::NodeShape>
{
public:
  ShapeInferenceAlgorithm(const loco::ShapeInferenceRule::Context *ctx) : _ctx{ctx}
  {
    // DO NOTHING
  }

private:
  const loco::ShapeInferenceRule::Context *_ctx;

private:
  bool shape_known(const loco::Node *node) const { return _ctx->known(node); }
  loco::NodeShape node_shape(const loco::Node *node) const { return _ctx->get(node); }

private:
  loco::NodeShape binary_node_shape(const moco::TFNode::Node *node)
  {
    // This helper works only for binary node.
    assert(node->arity() == 2);

    auto lhs_shape = node_shape(node->arg(0));
    auto rhs_shape = node_shape(node->arg(1));

    loco::TensorShape lhs_tensorshape = lhs_shape.as<loco::TensorShape>();
    loco::TensorShape rhs_tensorshape = rhs_shape.as<loco::TensorShape>();
    loco::TensorShape sum_tensorshape = moco::broadcast_shape(lhs_tensorshape, rhs_tensorshape);

    loco::NodeShape sum_shape({sum_tensorshape});

    return sum_shape;
  }

  loco::NodeShape node_shape_with_check(const moco::TFNode::Node *node)
  {
    auto nodeshape = node_shape(node);
    assert(nodeshape.domain() == loco::Domain::Tensor);

    return nodeshape;
  }

  bool valid_scalar_value(moco::TFConst *node)
  {
    auto nodeshape = node_shape(node);
    if (nodeshape.domain() != loco::Domain::Tensor)
    {
      return false;
    }
    if (node->dtype() != loco::DataType::S32)
    {
      return false;
    }

    auto tensor_shape = nodeshape.as<loco::TensorShape>();
    if (!(tensor_shape.rank() == 0 || tensor_shape.rank() == 1))
    {
      return false;
    }

    return true;
  }

  int32_t scalar_value(moco::TFConst *node)
  {
    auto nodeshape = node_shape(node);
    assert(node->dtype() == loco::DataType::S32);

    auto tensor_shape = nodeshape.as<loco::TensorShape>();
    assert(tensor_shape.rank() == 0 || tensor_shape.rank() == 1);

    return node->at<loco::DataType::S32>(0);
  }

public:
  loco::NodeShape visit(const moco::TFAdd *node) final { return binary_node_shape(node); }

  loco::NodeShape visit(const moco::TFAvgPool *node) final
  {
    auto value_shape = node_shape(node->value());
    assert(value_shape.domain() != loco::Domain::Unknown);

    moco::PlaneInference infer_plane_shape;

    infer_plane_shape.padding(node->padding());
    infer_plane_shape.stride(moco::stride_of(node->strides(), node->data_layout()));
    infer_plane_shape.window(moco::window_of(node->ksize(), node->data_layout()));

    auto input_feature_shape = moco::as_feature_shape(value_shape, node->data_layout());
    auto input_plane_shape = moco::make_plane_shape(input_feature_shape);
    auto output_feature_shape = input_feature_shape;
    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    moco::update(output_feature_shape).with(output_plane_shape);

    return moco::as_tensor_shape(output_feature_shape, node->data_layout());
  }

  loco::NodeShape visit(const moco::TFBiasAdd *node) final
  {
    return node_shape_with_check(node->value());
  }

  loco::NodeShape visit(const moco::TFConcatV2 *node) final
  {
    // axis shape should be available
    auto axis_node = node->axis();
    auto axis_shape = node_shape(axis_node);
    assert(axis_shape.domain() != loco::Domain::Unknown);

    // check all input shapes and all ranks should be same
    auto value_a = node->values(0);
    auto value_a_shape = node_shape(value_a);
    assert(value_a_shape.domain() == loco::Domain::Tensor);
    auto value_a_tensor_shape = value_a_shape.as<loco::TensorShape>();
    uint32_t a_rank = value_a_tensor_shape.rank();

    uint32_t num_values = node->num_values();
    for (uint32_t ni = 1; ni < num_values; ++ni)
    {
      auto value_b = node->values(ni);
      auto value_b_shape = node_shape(value_b);
      assert(value_b_shape.domain() == loco::Domain::Tensor);
      auto value_b_tensor_shape = value_b_shape.as<loco::TensorShape>();
      assert(a_rank == value_b_tensor_shape.rank());
    }

    int32_t axis_value = 0;
    bool axis_available = false;
    {
      // check for axis is TFConst
      auto tfconst = dynamic_cast<moco::TFConst *>(axis_node);
      if (tfconst != nullptr)
      {
        if (valid_scalar_value(tfconst))
        {
          axis_value = scalar_value(tfconst);
          axis_available = true;
        }
      }
    }
    if (!axis_available)
    {
      // TODO may need to refine error message
      throw oops::UserExn("ConcatV2 node does not have axis input", node->name());
    }

    uint32_t axis_absolute = (axis_value >= 0) ? axis_value : (int32_t)a_rank + axis_value;
    loco::TensorShape output_tensor_shape = value_a_tensor_shape;

    for (uint32_t index = 0; index < a_rank; ++index)
    {
      if (value_a_tensor_shape.dim(index).known())
      {
        uint32_t dim = value_a_tensor_shape.dim(index).value();
        uint32_t dim_acc = dim;

        for (uint32_t ni = 1; ni < num_values; ++ni)
        {
          auto value_b = node->values(ni);
          auto value_b_shape = node_shape(value_b);
          assert(value_b_shape.domain() == loco::Domain::Tensor);
          auto value_b_tensor_shape = value_b_shape.as<loco::TensorShape>();
          assert(value_b_tensor_shape.dim(index).known());
          if (index == axis_absolute)
            dim_acc += value_b_tensor_shape.dim(index).value();
          else
            assert(dim == value_b_tensor_shape.dim(index).value());
        }
        output_tensor_shape.dim(index) = dim_acc;
      }
      else
        output_tensor_shape.dim(index).unset();
    }
    return loco::NodeShape(output_tensor_shape);
  }

  loco::NodeShape visit(const moco::TFConst *node) final
  {
    loco::TensorShape output_tensor_shape;

    uint32_t rank = node->rank();
    output_tensor_shape.rank(rank);
    for (uint32_t index = 0; index < rank; ++index)
    {
      if (node->dim(index).known())
        output_tensor_shape.dim(index) = node->dim(index).value();
      else
        output_tensor_shape.dim(index).unset();
    }

    return loco::NodeShape(output_tensor_shape);
  }

  loco::NodeShape visit(const moco::TFConv2D *node) final
  {
    auto input_shape = moco::node_shape(node->input());
    auto ker_shape = moco::node_shape(node->filter());
    auto ker_tensor_shape = ker_shape.as<loco::TensorShape>(); // in HWIO
    auto node_stride = moco::stride_of(node->strides(), node->data_layout());
    auto node_window = moco::window_of(ker_tensor_shape, "HWIO");

    moco::PlaneInference infer_plane_shape;

    infer_plane_shape.padding(node->padding());
    infer_plane_shape.stride(node_stride);
    infer_plane_shape.window(node_window);

    auto input_feature_shape = moco::as_feature_shape(input_shape, node->data_layout());
    auto input_plane_shape = moco::make_plane_shape(input_feature_shape);
    // output count is from input count, depth is from kernel 'O' which is dim(3)
    auto output_feature_shape = input_feature_shape;
    output_feature_shape.depth() = ker_tensor_shape.dim(3).value();

    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    moco::update(output_feature_shape).with(output_plane_shape);

    return moco::as_tensor_shape(output_feature_shape, node->data_layout());
  }

  loco::NodeShape visit(const moco::TFConv2DBackpropInput *node) final
  {
    // TFConv2DBackpropInput's first input, named 'input_sizes', actually contains shape of node
    // output's feature map. We can get shape of TFConv2DBackpropInput by just copying this.
    // TODO Support when 'input_sizes' is not TFConst, or support constant folding
    auto input_sizes_node = dynamic_cast<moco::TFConst *>(node->input_sizes());
    if (input_sizes_node == nullptr)
    {
      // we are now supporting somekind of constant folding for this node, wait till it is finished
      loco::NodeShape unknown;
      return unknown;
    }

    // Let's support S32 for time being
    // TODO Support other integer types
    assert(input_sizes_node->dtype() == loco::DataType::S32);
    assert(input_sizes_node->size<loco::DataType::S32>() == 4);

    // copy!
    loco::TensorShape ofm_tensor_shape;
    ofm_tensor_shape.rank(4);
    for (uint32_t i = 0; i < 4; ++i)
    {
      int32_t dim = input_sizes_node->at<loco::DataType::S32>(i);
      assert(dim > 0);
      ofm_tensor_shape.dim(i) = (uint32_t)dim;
    }

    return loco::NodeShape(ofm_tensor_shape);
  }

  loco::NodeShape visit(const moco::TFDepthwiseConv2dNative *node) final
  {
    auto input_shape = moco::node_shape(node->input()); // NHWC
    auto ker_shape = moco::node_shape(node->filter());
    auto ker_tensor_shape = ker_shape.as<loco::TensorShape>(); // in HWCM
    auto node_stride = moco::stride_of(node->strides(), node->data_layout());
    auto node_window = moco::window_of(ker_tensor_shape, "HWCM");

    moco::PlaneInference infer_plane_shape;

    infer_plane_shape.padding(node->padding());
    infer_plane_shape.stride(node_stride);
    infer_plane_shape.window(node_window);

    auto input_feature_shape = moco::as_feature_shape(input_shape, node->data_layout());
    auto input_plane_shape = moco::make_plane_shape(input_feature_shape);
    // output count is from input count, depth is from kernel 'CM' which is dim(2) * dim(3)
    auto output_feature_shape = input_feature_shape;
    output_feature_shape.depth() =
      loco::Dimension(ker_tensor_shape.dim(2).value() * ker_tensor_shape.dim(3).value());

    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    moco::update(output_feature_shape).with(output_plane_shape);

    return moco::as_tensor_shape(output_feature_shape, node->data_layout());
  }

  loco::NodeShape visit(const moco::TFFakeQuantWithMinMaxVars *node) final
  {
    return node_shape_with_check(node->inputs());
  }

  loco::NodeShape visit(const moco::TFFusedBatchNorm *node) final
  {
    return node_shape_with_check(node->x());
  }

  loco::NodeShape visit(const moco::TFIdentity *node) final
  {
    return node_shape_with_check(node->input());
  }

  loco::NodeShape visit(const moco::TFMaximum *node) final { return binary_node_shape(node); }

  loco::NodeShape visit(const moco::TFMaxPool *node) final
  {
    auto input_shape = node_shape(node->input());
    assert(input_shape.domain() != loco::Domain::Unknown);

    moco::PlaneInference infer_plane_shape;

    infer_plane_shape.padding(node->padding());
    infer_plane_shape.stride(moco::stride_of(node->strides(), node->data_layout()));
    infer_plane_shape.window(moco::window_of(node->ksize(), node->data_layout()));

    auto input_feature_shape = moco::as_feature_shape(input_shape, node->data_layout());
    auto input_plane_shape = moco::make_plane_shape(input_feature_shape);
    auto output_feature_shape = input_feature_shape;
    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    moco::update(output_feature_shape).with(output_plane_shape);

    return moco::as_tensor_shape(output_feature_shape, node->data_layout());
  }

  loco::NodeShape visit(const moco::TFMean *node) final
  {
    auto input_shape = node_shape(node->input());
    auto reduction_indices = node->reduction_indices();

    // Get constant values if reduction_indices is const
    std::vector<int32_t> reduction_values;
    if (auto tfconst = dynamic_cast<moco::TFConst *>(reduction_indices))
    {
      assert(tfconst->dtype() == loco::DataType::S32);
      auto const_size = tfconst->size<loco::DataType::S32>();
      for (uint32_t i = 0; i < const_size; ++i)
      {
        int32_t axis = tfconst->at<loco::DataType::S32>(i);
        if (axis < 0)
          axis += input_shape.as<loco::TensorShape>().rank();
        reduction_values.push_back(axis);
      }
    }
    else
    {
      // we cannot find a valid reduction indices value
      loco::NodeShape unknown;
      return unknown;
    }

    loco::TensorShape output_shape;
    auto input_tensor_shape = input_shape.as<loco::TensorShape>();

    if (node->keep_dims())
    {
      output_shape.rank(input_tensor_shape.rank());
      for (uint32_t i = 0; i < input_tensor_shape.rank(); ++i)
        output_shape.dim(i) = input_tensor_shape.dim(i);
      for (uint32_t i = 0; i < reduction_values.size(); ++i)
        output_shape.dim(reduction_values.at(i)) = 1;
    }
    else
    {
      std::vector<bool> check_reduce(input_tensor_shape.rank(), false);
      for (uint32_t i = 0; i < reduction_values.size(); ++i)
        check_reduce.at(reduction_values.at(i)) = true;

      uint32_t reduce_cnt = 0;
      for (uint32_t i = 0; i < check_reduce.size(); ++i)
        if (check_reduce.at(i))
          ++reduce_cnt;

      output_shape.rank(input_tensor_shape.rank() - reduce_cnt);
      for (uint32_t i = 0, j = 0; i < check_reduce.size(); ++i)
        if (check_reduce.at(i) == false)
          output_shape.dim(j++) = i;
    }

    return loco::NodeShape(output_shape);
  }

  loco::NodeShape visit(const moco::TFMul *node) final { return binary_node_shape(node); }

  loco::NodeShape visit(const moco::TFPack *node) final
  {
    loco::NodeShape unknown;

    auto input_shape_0 = node_shape(node->values(0));
    if (input_shape_0.domain() != loco::Domain::Tensor)
    {
      // TODO fix this for other cases
      // We support only valid tensor shape for now
      return unknown;
    }
    loco::TensorShape tensor_shape_0 = input_shape_0.as<loco::TensorShape>();

    // all input shapes should be same
    auto num_values = node->N();
    for (uint32_t i = 1; i < num_values; ++i)
    {
      auto input_shape = node_shape(node->values(i));
      if (input_shape.domain() != loco::Domain::Tensor)
      {
        // TODO ditto
        return unknown;
      }

      loco::TensorShape tensor_shape = input_shape.as<loco::TensorShape>();
      if (!(input_shape_0 == input_shape))
      {
        throw oops::UserExn("All input values shape should be same", node->name());
      }
    }

    // output rank will be +1 of rank of the input
    // axis should be in range of [-r, r), where r is rank of the output
    auto axis = node->axis();
    int32_t rank = static_cast<int32_t>(tensor_shape_0.rank());
    assert(rank >= 0);
    int32_t rank_output = rank + 1;
    if (axis < -rank_output || rank_output <= axis)
    {
      throw oops::UserExn("axis is out of range", node->name());
    }

    auto axis_stack = (axis >= 0) ? axis : rank_output + axis;

    loco::TensorShape output_tensor_shape;

    output_tensor_shape.rank(rank_output);
    for (int32_t r = 0; r < axis_stack; ++r)
    {
      output_tensor_shape.dim(r).set(tensor_shape_0.dim(r).value());
    }
    output_tensor_shape.dim(axis_stack).set(num_values);
    for (int32_t r = axis_stack; r < rank; ++r)
    {
      output_tensor_shape.dim(r + 1).set(tensor_shape_0.dim(r).value());
    }

    return loco::NodeShape(output_tensor_shape);
  }

  loco::NodeShape visit(const moco::TFPad *node) final
  {
    auto input_shape = node_shape(node->input());
    assert(input_shape.domain() == loco::Domain::Tensor);

    auto const_paddings = loco::must_cast<moco::TFConst *>(node->paddings());
    assert(const_paddings->dtype() == loco::DataType::S32);
    assert(const_paddings->rank() == 2);

    loco::TensorShape input_tensor_shape = input_shape.as<loco::TensorShape>();
    loco::TensorShape output_tensor_shape;

    output_tensor_shape.rank(input_tensor_shape.rank());
    for (uint32_t axis = 0; axis < input_tensor_shape.rank(); ++axis)
    {
      output_tensor_shape.dim(axis) = input_tensor_shape.dim(axis).value() +
                                      const_paddings->at<loco::DataType::S32>(axis * 2) +
                                      const_paddings->at<loco::DataType::S32>(axis * 2 + 1);
    }

    return loco::NodeShape{output_tensor_shape};
  }

  loco::NodeShape visit(const moco::TFPlaceholder *node) final
  {
    loco::TensorShape output_tensor_shape;

    uint32_t rank = node->rank();
    output_tensor_shape.rank(rank);
    for (uint32_t index = 0; index < rank; ++index)
    {
      if (node->dim(index).known())
        output_tensor_shape.dim(index) = node->dim(index).value();
      else
        output_tensor_shape.dim(index).unset();
    }

    return loco::NodeShape(output_tensor_shape);
  }

  loco::NodeShape visit(const moco::TFRealDiv *node) final { return binary_node_shape(node); }

  loco::NodeShape visit(const moco::TFRelu *node) final
  {
    return node_shape_with_check(node->features());
  }

  loco::NodeShape visit(const moco::TFRelu6 *node) final
  {
    return node_shape_with_check(node->features());
  }

  loco::NodeShape visit(const moco::TFReshape *node) final
  {
    loco::NodeShape unknown;

    // For now, we only consider Fixed Reshape, i.e. Reshape with determined
    //      'shape' input. So here we only support case when 'shape' input of
    //      TFReshape is TFConst. If 'shape' input is not TFConst, another
    //      transform (e.g. constant folding) should be done beforehand to make
    //      it TFConst.
    // TODO Support dynamic Reshape
    // Note that 'shape()' here is 'shape' input, not node's shape information
    auto const_shape_input = dynamic_cast<moco::TFConst *>(node->shape());
    if (!const_shape_input)
    {
      // 'shape' input of TFReshape is not TFConst, we can not do shape inference
      return unknown;
    }

    // 'Shape' input should be integer tensor of rank 1, e.g. [2, 3, 4] or [3, -1]
    assert(const_shape_input->dtype() == loco::DataType::S32);
    assert(const_shape_input->rank() == 1);

    auto shape_rank = const_shape_input->dim(0).value();
    assert(shape_rank > 0);

    loco::TensorShape output_shape;
    output_shape.rank(shape_rank);
    for (uint32_t axis = 0; axis < shape_rank; ++axis)
    {
      auto shape_dim = const_shape_input->at<loco::DataType::S32>(axis);
      if (shape_dim == -1)
      {
        // Reshape's new shape has wildcard dimension, i.e. dynamic reshape
        return unknown;
      }
      assert(shape_dim >= 1);
      output_shape.dim(axis) = shape_dim;
    }

    // TODO Compare 'tensor' input and validate coherency?
    // Not sure this is appropriate stage for this task.

    return loco::NodeShape(output_shape);
  }

  loco::NodeShape visit(const moco::TFRsqrt *node) final
  {
    return node_shape_with_check(node->x());
  }

  loco::NodeShape visit(const moco::TFShape *node) final
  {
    auto input_shape = node_shape(node->input());
    auto input_tensor_shape = input_shape.as<loco::TensorShape>();

    loco::TensorShape output_shape;

    // Note that input shape becomes node(TFShape)'s value
    output_shape.rank(1);
    output_shape.dim(0) = input_tensor_shape.rank();

    return loco::NodeShape(output_shape);
  }

  loco::NodeShape visit(const moco::TFSoftmax *node) final
  {
    return node_shape_with_check(node->logits());
  }

  loco::NodeShape visit(const moco::TFSqrt *node) final { return node_shape_with_check(node->x()); }

  loco::NodeShape visit(const moco::TFSquaredDifference *node) final
  {
    return binary_node_shape(node);
  }

  loco::NodeShape visit(const moco::TFSqueeze *node) final
  {
    auto input_shape = node_shape(node->input());

    // TODO Not sure Squeeze only get input as Tensor
    // Note that tensor_shape() has assertion in it
    auto input_tensor_shape = input_shape.as<loco::TensorShape>();

    auto squeeze_dims_vec = node->squeeze_dims();
    std::set<int64_t> squeeze_dims(squeeze_dims_vec.cbegin(), squeeze_dims_vec.cend());

    loco::TensorShape output_shape;
    uint32_t output_rank = 0;

    if (squeeze_dims.empty())
    {
      // Remove all dimensions whose value is 1
      for (uint32_t axis = 0; axis < input_tensor_shape.rank(); ++axis)
      {
        assert(input_tensor_shape.dim(axis).known());
        auto dim = input_tensor_shape.dim(axis).value();
        if (dim != 1)
        {
          assert(dim > 1);
          output_shape.rank(++output_rank);
          output_shape.dim(output_rank - 1) = dim;
        }
      }
    }
    else
    {
      uint32_t input_rank = input_tensor_shape.rank();

      // Sanity check for 'squeeze_dims'
      auto is_valid_squeeze_dims = [&squeeze_dims, &input_rank]() {
        if (!(squeeze_dims.size() < input_rank))
          return false;
        for (auto squeeze_dim : squeeze_dims)
        {
          if (!(squeeze_dim >= -(int64_t)input_rank))
            return false;
          if (!(squeeze_dim < (int64_t)input_rank))
            return false;
        }
        return true;
      };

      if (!is_valid_squeeze_dims())
      {
        throw oops::UserExn("Invalid squeeze dimension", node->name());
      }

      // Resolve negative squeeze dimension
      std::set<int64_t> resolved_squeeze_dims;
      for (auto squeeze_dim : squeeze_dims)
      {
        if (squeeze_dim < 0)
          resolved_squeeze_dims.insert(squeeze_dim + (int64_t)input_rank);
        else
          resolved_squeeze_dims.insert(squeeze_dim);
      }

      // Remove squeeze dimensions only
      for (uint32_t axis = 0; axis < input_rank; ++axis)
      {
        assert(input_tensor_shape.dim(axis).known());
        auto dim = input_tensor_shape.dim(axis).value();
        if (resolved_squeeze_dims.find((int64_t)axis) == resolved_squeeze_dims.cend())
        {
          // Not squeeze dim
          output_shape.rank(++output_rank);
          output_shape.dim(output_rank - 1) = dim;
        }
        else
        {
          // Is squeeze dim
          assert(dim == 1);
          // DO NOTHING
        }
      }
    }

    assert(output_shape.rank() > 0);

    return loco::NodeShape(output_shape);
  }

  loco::NodeShape visit(const moco::TFStopGradient *node) final
  {
    return node_shape_with_check(node->input());
  }

  loco::NodeShape visit(const moco::TFStridedSlice *node) final
  {
    loco::NodeShape unknown;
    auto input_shape = node_shape(node->input());
    if (input_shape.domain() != loco::Domain::Tensor)
    {
      // TODO fix this for other cases
      // We support only tensor shape for now
      return unknown;
    }

    // TODO support full mask features: see import codes also
    // Limited attributes for now
    assert(node->begin_mask() == 0);
    assert(node->end_mask() == 0);
    assert(node->ellipsis_mask() == 0);
    assert(node->shrink_axis_mask() == 1);

    auto const_begin = loco::must_cast<moco::TFConst *>(node->begin());
    auto const_end = loco::must_cast<moco::TFConst *>(node->end());
    auto const_strides = loco::must_cast<moco::TFConst *>(node->strides());

    assert(dynamic_cast<moco::TFConst *>(node->input()) != nullptr);
    assert(const_begin != nullptr);
    assert(const_end != nullptr);
    assert(const_strides != nullptr);

    auto input_tensor_shape = input_shape.as<loco::TensorShape>();
    auto input_rank = input_tensor_shape.rank();
    auto output_rank = input_rank;

    // TODO support strides with > 1
    uint32_t elements = const_strides->size<loco::DataType::S32>();
    for (uint32_t e = 0; e < elements; ++e)
      assert(const_strides->at<loco::DataType::S32>(e) == 1);

    // lets apply begin ~ end range from input shape
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

    // get final tensor shape from applying shrink mask to output_shape_range
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

    return loco::NodeShape(output_tensor_shape);
  }

  loco::NodeShape visit(const moco::TFSub *node) final { return binary_node_shape(node); }

  loco::NodeShape visit(const moco::TFTanh *node) final { return node_shape_with_check(node->x()); }

  // For virtual nodes
  loco::NodeShape visit(const moco::TFPush *node) { return node_shape_with_check(node->from()); }

public:
  loco::NodeShape visit(const moco::TFNode *) final
  {
    loco::NodeShape unknown;
    return unknown;
  }
};

} // namespace

namespace
{
namespace compat
{

struct Context final : public loco::ShapeInferenceRule::Context
{
  bool known(const loco::Node *node) const final { return loco::shape_known(node); }
  loco::NodeShape get(const loco::Node *node) const final { return loco::shape_get(node); }
};

class Sink final : public loco::ShapeInferenceRule::Sink
{
public:
  enum Status
  {
    Unknown,
    Okay,
    Fail,
  };

public:
  const Status &status(void) const { return _status; }
  const loco::NodeShape &shape(void) const { return _shape; }

public:
  void okay(const loco::NodeShape &shape) final
  {
    _status = Okay;
    _shape = shape;
  }

  void fail(void) final
  {
    // Notify failrue
    _status = Fail;
  }

private:
  Status _status = Unknown;
  loco::NodeShape _shape;
};

} // namespace compat
} // namespace

namespace moco
{

bool TFShapeInferenceRule::support(const API &api) const
{
  return api == API::V1 or api == API::V2;
}

bool TFShapeInferenceRule::recognize(const loco::Dialect *d) const
{
  // handle only TensorFlow dialect
  return TFDialect::get() == d;
}

bool TFShapeInferenceRule::infer(const loco::Node *node, loco::NodeShape &shape) const
{
  ::compat::Context ctx;
  ::compat::Sink sink;

  infer(&ctx, node, &sink);

  assert(sink.status() == ::compat::Sink::Okay or sink.status() == ::compat::Sink::Fail);

  if (sink.status() == ::compat::Sink::Fail)
  {
    return false;
  }

  shape = sink.shape();

  return true;
}

void TFShapeInferenceRule::infer(const Context *ctx, const loco::Node *node, Sink *sink) const
{
  assert(node->dialect() == TFDialect::get());
  assert(dynamic_cast<const TFNode *>(node) != nullptr);

  ShapeInferenceAlgorithm alg{ctx};
  auto shape = loco::must_cast<const TFNode *>(node)->accept(&alg);

  if (shape.domain() == loco::Domain::Unknown)
    sink->fail();
  else
    sink->okay(shape);
}

} // namespace moco
