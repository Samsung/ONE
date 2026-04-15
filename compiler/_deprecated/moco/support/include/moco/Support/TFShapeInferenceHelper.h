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

#ifndef __MOCO_SUPPORT_SHAPE_INFERENCE_HELPER_H__
#define __MOCO_SUPPORT_SHAPE_INFERENCE_HELPER_H__

#include <moco/IR/TFDataLayout.h>
#include <moco/IR/TFPadding.h>

#include <loco.h>
#include <loco/IR/NodeShape.h>
#include <loco/IR/Padding2D.h>
#include <loco/IR/Stride.h>
#include <loco/IR/Window.h>

#include <cassert>

namespace moco
{

/**
 * @note  Helper for return broadcasted shape for binary operators having
 *        different shape for input x and y
 */
loco::TensorShape broadcast_shape(const loco::TensorShape &x, const loco::TensorShape &y);

} // namespace moco

namespace moco
{

/**
 * @brief  Return true if node has shape inference data for checking shape
 *         inference is done or not
 *
 * @note   Will be deprecated in near future
 */
bool shape_inference_done(const loco::Node *node);

/**
 * @note  While in shape inference, Node maybe Canonical, TF dialect or other dialects
 *        This will provide common loco::NodeShape as shape information
 */
loco::NodeShape node_shape(const loco::Node *node);
bool node_shape(const loco::Node *node, loco::NodeShape &nodeshape);

loco::TensorShape as_tensor_shape(const loco::FeatureShape &feature_shape,
                                  const TFDataLayout &data_layout);

loco::FeatureShape as_feature_shape(const loco::NodeShape &nodeshape,
                                    const TFDataLayout &data_layout);

} // namespace moco

namespace moco
{

struct PlaneShape
{
  loco::Dimension height;
  loco::Dimension width;
};

class FeatureShapeUpdater final
{
public:
  FeatureShapeUpdater(loco::FeatureShape *ptr) : _feature_shape_ptr{ptr}
  {
    // DO NOTHING
  }

public:
  void with(const PlaneShape &plane_shape) const
  {
    _feature_shape_ptr->height() = plane_shape.height;
    _feature_shape_ptr->width() = plane_shape.width;
  }

private:
  loco::FeatureShape *_feature_shape_ptr;
};

PlaneShape make_plane_shape(const loco::FeatureShape &feature_shape);

FeatureShapeUpdater update(loco::FeatureShape &feature_shape);

class PlaneInference
{
protected:
  struct Parameters
  {
    PlaneShape input;
    PlaneShape stride;
    PlaneShape window;
    PlaneShape dilation;
    PlaneShape effective_window;
    PlaneShape output;
  };

  void fill(Parameters &p, const PlaneShape &in)
  {
    p.input.height = in.height;
    p.input.width = in.width;

    p.stride.height = _stride.vertical();
    p.stride.width = _stride.horizontal();

    p.window.height = _window.vertical();
    p.window.width = _window.horizontal();

    // TODO support dilation
    p.dilation.height = 1;
    p.dilation.width = 1;

    p.effective_window.height = p.dilation.height.value() * (p.window.height.value() - 1) + 1;
    p.effective_window.width = p.dilation.width.value() * (p.window.width.value() - 1) + 1;
  }

  PlaneShape infer(const Parameters &p, const PlaneShape &)
  {
    PlaneShape res;

    if (_padding == "VALID")
    {
      res.height =
        (p.input.height.value() + p.stride.height.value() - p.effective_window.height.value()) /
        p.stride.height.value();
      res.width =
        (p.input.width.value() + p.stride.width.value() - p.effective_window.width.value()) /
        p.stride.width.value();
    }
    else if (_padding == "SAME")
    {
      res.height = (p.input.height.value() + p.stride.height.value() - 1) / p.stride.height.value();
      res.width = (p.input.width.value() + p.stride.width.value() - 1) / p.stride.width.value();
    }
    else
      assert(false);

    return res;
  }

public:
  PlaneShape operator()(const PlaneShape &in)
  {
    Parameters p;

    fill(p, in);

    return infer(p, in);
  }

public:
  void padding(const TFPadding &value) { _padding = value; }
  void window(const loco::Window<2> value) { _window = value; }
  void stride(const loco::Stride<2> value) { _stride = value; }

private:
  TFPadding _padding;
  loco::Window<2> _window;
  loco::Stride<2> _stride;
};

class Padding2DInference final : public PlaneInference
{
public:
  loco::Padding2D operator()(const PlaneShape &in)
  {
    Parameters p;

    fill(p, in);

    auto output = infer(p, in);

    int64_t i_height = (int64_t)(output.height.value() - 1) * (int64_t)p.stride.height.value() +
                       (int64_t)p.effective_window.height.value() - (int64_t)p.input.height.value();
    int64_t i_width = (int64_t)(output.width.value() - 1) * (int64_t)p.stride.width.value() +
                      (int64_t)p.effective_window.width.value() - (int64_t)p.input.width.value();

    uint32_t pad_height = i_height >= 0 ? (uint32_t)i_height : 0U;
    uint32_t pad_width = i_width >= 0 ? (uint32_t)i_width : 0U;

    loco::Padding2D padding2d;

    padding2d.top(pad_height / 2);
    padding2d.bottom(pad_height - padding2d.top());
    padding2d.left(pad_width / 2);
    padding2d.right(pad_width - padding2d.left());

    return padding2d;
  }
};

} // namespace moco

namespace moco
{

using TFStrides = std::vector<int64_t>;
using TFKSize = std::vector<int64_t>;

loco::Stride<2> stride_of(const TFStrides &strides, const TFDataLayout &datalayout);
loco::Window<2> window_of(const TFKSize &ksize, const TFDataLayout &datalayout);
loco::Window<2> window_of(const loco::TensorShape &shape, const TFDataLayout &datalayout);

} // namespace moco

#endif // __MOCO_SERVICE_SHAPE_INFERENCE_HELPER_H__
