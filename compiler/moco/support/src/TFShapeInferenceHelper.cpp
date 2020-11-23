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

#include "moco/Support/TFShapeInferenceHelper.h"

#include <loco/Service/ShapeInference.h>

#include <oops/InternalExn.h>

#include <cassert>

namespace
{

// TODO Use codes in loco and remove duplicate broadcast_shape() and related
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
    {
      // TODO may need to refine message
      INTERNAL_EXN("ShapeInference: Input shapes don't match");
    }

    output_shape.dim(axis) = std::max(x_dim, y_dim);
  }

  return output_shape;
}

} // namespace

namespace moco
{

loco::TensorShape broadcast_shape(const loco::TensorShape &x, const loco::TensorShape &y)
{
  auto x_match = x;
  auto y_match = y;

  expand_rank(x_match, y_match);

  auto output_shape = expand_dimension(x_match, y_match);

  return output_shape;
}

} // namespace moco

namespace moco
{

loco::NodeShape node_shape(const loco::Node *node)
{
  loco::NodeShape nodeshape; // default domain is Unknown

  if (loco::shape_known(node))
  {
    nodeshape = loco::shape_get(node);
  }

  return nodeshape;
}

bool node_shape(const loco::Node *node, loco::NodeShape &nodeshape)
{
  nodeshape = node_shape(node);
  return (nodeshape.domain() != loco::Domain::Unknown);
}

loco::TensorShape as_tensor_shape(const loco::FeatureShape &feature_shape,
                                  const TFDataLayout &data_layout)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(4);
  if (data_layout == "NHWC")
  {
    tensor_shape.dim(0) = feature_shape.count();
    tensor_shape.dim(1) = feature_shape.height();
    tensor_shape.dim(2) = feature_shape.width();
    tensor_shape.dim(3) = feature_shape.depth();
  }
  else if (data_layout == "NCHW")
  {
    tensor_shape.dim(0) = feature_shape.count();
    tensor_shape.dim(1) = feature_shape.depth();
    tensor_shape.dim(2) = feature_shape.height();
    tensor_shape.dim(3) = feature_shape.width();
  }
  else
  {
    // TODO support for other data_layout if needed
    INTERNAL_EXN_V("ShapeInference: Unknown data_format", data_layout);
  }

  return tensor_shape;
}

loco::FeatureShape as_feature_shape(const loco::NodeShape &nodeshape,
                                    const TFDataLayout &data_layout)
{
  if (nodeshape.domain() == loco::Domain::Feature)
    return nodeshape.as<loco::FeatureShape>();

  loco::FeatureShape feature_shape;

  // only convert from tensor to feature
  if (nodeshape.domain() != loco::Domain::Tensor)
  {
    INTERNAL_EXN("ShapeInference: Invalid shape information");
  }

  loco::TensorShape tensor_shape = nodeshape.as<loco::TensorShape>();

  if (tensor_shape.rank() != 4)
  {
    INTERNAL_EXN("ShapeInference: Rank is not 4");
  }

  if (data_layout == "NHWC")
  {
    feature_shape.count() = tensor_shape.dim(0);
    feature_shape.height() = tensor_shape.dim(1);
    feature_shape.width() = tensor_shape.dim(2);
    feature_shape.depth() = tensor_shape.dim(3);
  }
  else if (data_layout == "NCHW")
  {
    feature_shape.count() = tensor_shape.dim(0);
    feature_shape.depth() = tensor_shape.dim(1);
    feature_shape.height() = tensor_shape.dim(2);
    feature_shape.width() = tensor_shape.dim(3);
  }
  else
  {
    // TODO support for other data_layout if needed
    INTERNAL_EXN_V("ShapeInference: Unknown data_format", data_layout);
  }

  return feature_shape;
}

} // namespace moco

namespace moco
{

PlaneShape make_plane_shape(const loco::FeatureShape &feature_shape)
{
  PlaneShape plane_shape;

  plane_shape.height = feature_shape.height();
  plane_shape.width = feature_shape.width();

  return plane_shape;
}

FeatureShapeUpdater update(loco::FeatureShape &feature_shape)
{
  return FeatureShapeUpdater{&feature_shape};
}

} // namespace moco

namespace
{

/**
 * @brief Class to represent TensorFlow "data_format" attr.
 */
enum class DataLayout
{
  NHWC,
  NCHW,
};

DataLayout as_data_layout(const std::string &tf_layout_str)
{
  if (tf_layout_str == "NHWC")
    return DataLayout::NHWC;
  else if (tf_layout_str == "NCHW")
    return DataLayout::NCHW;
  else
    /// @note data layout tag in TensorFlow is 'data_format'
    INTERNAL_EXN_V("ShapeInference: Unknown data_format", tf_layout_str);
}

} // namespace

namespace moco
{

loco::Stride<2> stride_of(const TFStrides &strides, const TFDataLayout &datalayout)
{
  loco::Stride<2> stride;

  auto data_layout = as_data_layout(datalayout);
  if (data_layout == DataLayout::NHWC)
  {
    stride.vertical(strides[1]);
    stride.horizontal(strides[2]);
  }
  else if (data_layout == DataLayout::NCHW)
  {
    stride.vertical(strides[2]);
    stride.horizontal(strides[3]);
  }
  else
  {
    // TODO add more datalayout supports if needed
    INTERNAL_EXN("ShapeInference: Unknown data_format");
  }

  return stride;
}

loco::Window<2> window_of(const TFKSize &ksize, const TFDataLayout &datalayout)
{
  loco::Window<2> window;

  auto data_layout = as_data_layout(datalayout);
  if (data_layout == DataLayout::NHWC)
  {
    window.vertical(ksize[1]);
    window.horizontal(ksize[2]);
  }
  else if (data_layout == DataLayout::NCHW)
  {
    window.vertical(ksize[2]);
    window.horizontal(ksize[3]);
  }
  else
  {
    // TODO add more datalayout supports if needed
    INTERNAL_EXN("ShapeInference: Unknown data_format");
  }

  return window;
}

loco::Window<2> window_of(const loco::TensorShape &shape, const TFDataLayout &datalayout)
{
  loco::Window<2> window;

  if (datalayout == "HWIO")
  {
    window.vertical(shape.dim(0).value());
    window.horizontal(shape.dim(1).value());
  }
  else if (datalayout == "HWCM")
  {
    window.vertical(shape.dim(0).value());
    window.horizontal(shape.dim(1).value());
  }
  else
  {
    // TODO add more datalayout supports if needed
    INTERNAL_EXN_V("ShapeInference: Unknown data_format", datalayout);
  }

  return window;
}

} // namespace moco
