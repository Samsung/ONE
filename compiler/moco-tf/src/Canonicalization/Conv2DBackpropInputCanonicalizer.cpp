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

#include "Conv2DBackpropInputCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include "CodecHelper.h"

#include <loco/IR/Stride.h>
#include <loco/IR/Padding2D.h>
#include <loco/Service/ShapeInference.h>

#include <oops/UserExn.h>

namespace
{
using plier::tf::DataLayout;

void set_filter_enc(loco::FilterEncode *filter_enc)
{
  auto enc = std::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>();

  // In TensorFlow, Conv2dBackpropInput's filter is a 4-D tensor of following shape:
  // [filter_height, filter_width, out_channels, in_channels] or HWOI or HWNC (in/out in loco sense)
  enc->perm()->axis(loco::FilterAxis::Height) = 0;
  enc->perm()->axis(loco::FilterAxis::Width) = 1;
  enc->perm()->axis(loco::FilterAxis::Count) = 2;
  enc->perm()->axis(loco::FilterAxis::Depth) = 3;

  filter_enc->encoder(std::move(enc));
}

} // namespace

namespace
{

bool stride_2d_from_4d(loco::Stride<2> &ret, const std::vector<int64_t> &strides_4d,
                       const DataLayout data_layout)
{
  if (!(strides_4d.size() == 4))
    return false;

  switch (data_layout)
  {
    case DataLayout::NHWC:
      ret.vertical(strides_4d.at(1));
      ret.horizontal(strides_4d.at(2));
      break;
    case DataLayout::NCHW:
      ret.vertical(strides_4d.at(2));
      ret.horizontal(strides_4d.at(3));
      break;
    default:
      return false;
  }
  return true;
}

struct PlaneShape
{
  loco::Dimension vertical;
  loco::Dimension horizontal;
};

class Padding2DInference final
{
public:
  Padding2DInference(const moco::TFNode *node) { _node = node; }

public:
  loco::Padding2D operator()(void);

public:
  PlaneShape &input() { return _input; }
  PlaneShape &output() { return _output; }
  loco::Stride<2> &stride() { return _stride; }
  loco::Window<2> &window() { return _window; }
  moco::TFPadding &padding() { return _padding; }

private:
  /// @brief  Check whether ingredients set by non-default values
  bool ready()
  {
    if (not input().vertical.known())
      return false;
    if (not input().horizontal.known())
      return false;
    if (not output().vertical.known())
      return false;
    if (not output().horizontal.known())
      return false;
    if (stride().vertical() == 0)
      return false;
    if (stride().horizontal() == 0)
      return false;
    if (window().vertical() == 0)
      return false;
    if (window().horizontal() == 0)
      return false;
    if (padding().empty())
      return false;

    return true;
  }

  inline uint32_t tight_output_for_valid_padding(uint32_t input, uint32_t stride, uint32_t filter)
  {
    return stride * (input - 1) + filter;
  }

  /**
   * @note  For Conv2DBackpropInput SAME padding, TensorFlow requires this condition to hold
   *
   * Reference: `::tensorflow::GetWindowedOutputSizeVerboseV2()` from TensorFlow project
   */
  inline bool same_padding_applicable(uint32_t input, uint32_t output, uint32_t stride)
  {
    // Here 'input' and 'output' means Conv2DBackpropInput's actual node input and output.
    // Then these three conditions are equivalent:
    //
    //     input == floor((output + stride - 1) / stride)
    //     input == ceil(output / stride)
    //     (stride * (input - 1) < output) and (output <= stride * input)
    return (stride * (input - 1) < output) and (output <= stride * input);
  }

  inline uint32_t padding_needed(uint32_t input, uint32_t output, uint32_t stride, uint32_t filter)
  {
    return stride * (input - 1) + filter - output;
  }

private:
  const moco::TFNode *_node;
  PlaneShape _input;
  PlaneShape _output;
  loco::Stride<2> _stride;
  loco::Window<2> _window;
  moco::TFPadding _padding;
};

loco::Padding2D Padding2DInference::operator()(void)
{
  assert(ready());

  if (padding() == "VALID")
  {
    // In case of VALID padding, TensorFlow accepts any size same or larger than
    // 'tight fit' output. When output size (set by 'input sizes' node input) is
    // larger than tight fit, extra spaces filled with zero.
    auto tight_output_vertical = tight_output_for_valid_padding(
      input().vertical.value(), stride().vertical(), window().vertical());
    auto tight_output_horizontal = tight_output_for_valid_padding(
      input().horizontal.value(), stride().horizontal(), window().horizontal());

    if (output().vertical.value() < tight_output_vertical or
        output().horizontal.value() < tight_output_horizontal)
      throw oops::UserExn("input_sizes is too small", _node->name());

    // Currently, only accept tight fit.
    // TODO Support non-tight case by adding zero padding operation
    assert(output().vertical.value() == tight_output_vertical);
    assert(output().horizontal.value() == tight_output_horizontal);

    return loco::Padding2D(0, 0, 0, 0);
  }

  if (padding() == "SAME")
  {
    // This condition is required by TensorFlow
    if (not same_padding_applicable(input().vertical.value(), output().vertical.value(),
                                    stride().vertical()) or
        not same_padding_applicable(input().horizontal.value(), output().horizontal.value(),
                                    stride().horizontal()))
      throw oops::UserExn("Size mismatch for SAME padding", _node->name());

    auto whole_pad_vertical = padding_needed(input().vertical.value(), output().vertical.value(),
                                             stride().vertical(), window().vertical());
    auto whole_pad_horizontal =
      padding_needed(input().horizontal.value(), output().horizontal.value(), stride().horizontal(),
                     window().horizontal());

    loco::Padding2D res;

    res.top(whole_pad_vertical / 2);
    res.bottom(whole_pad_vertical - res.top());
    res.left(whole_pad_horizontal / 2);
    res.right(whole_pad_horizontal - res.left());

    return res;
  }

  throw oops::UserExn("Usupported padding " + padding(), _node->name());
}

/**
 * @param[out] ret  PlaneShape extracted from 'node' with given 'data_layout'
 * @param[in]  node
 * @param[in]  data_layout
 *
 * @return true on success
 */
bool set_plane_shape(PlaneShape &ret, const loco::Node *node, const DataLayout data_layout)
{
  auto tensor_shape = loco::shape_get(node).as<loco::TensorShape>();
  if (!(tensor_shape.rank() == 4))
    return false;

  switch (data_layout)
  {
    case DataLayout::NHWC:
      ret.vertical = tensor_shape.dim(1).value();
      ret.horizontal = tensor_shape.dim(2).value();
      break;
    case DataLayout::NCHW:
      ret.vertical = tensor_shape.dim(2).value();
      ret.horizontal = tensor_shape.dim(3).value();
      break;
    default:
      return false;
  }

  return true;
}

/**
 * @param[out] ret  2D Window extracted from HW** filter node
 * @param[in]  filter_node
 *
 * @return true on success
 */
bool set_window(loco::Window<2> &ret, const loco::Node *filter_node)
{
  auto tensor_shape = loco::shape_get(filter_node).as<loco::TensorShape>();
  assert(tensor_shape.rank() == 4);

  ret.vertical(tensor_shape.dim(0).value());
  ret.horizontal(tensor_shape.dim(1).value());

  return true;
}

} // namespace

namespace
{

bool canonicalize_conv2d_backprop_input(loco::Graph *graph,
                                        moco::TFConv2DBackpropInput *conv2d_backprop)
{
  /**
   * @note This will replace TFConv2DBackpropInput node with canonical
   *       FeatureEncode + FilterEncode + TransposedConv2D + FeatureDecode
   *
   * Before
   *           input_sizes ----
   *                           \
   *           filter -------- TFConv2DBackpropInput --- output(s)
   *                           /
   *           out_backprop ---
   *
   * After
   *           input_sizes ----
   *                           \
   *           filter -------- TFConv2DBackpropInput ---
   *                           /
   *           out_backprop ---
   *
   *           filter ------ FilterEncode ------ TransposedConv2D --- FeatureDecode --- output(s)
   *                          (as ker)           /
   *           out_backprop --- FeatureEncode ---
   *                             (as ifm)
   */

  if (!loco::shape_known(conv2d_backprop->out_backprop()))
    return false;
  if (!loco::shape_known(conv2d_backprop))
    return false;
  if (!loco::shape_known(conv2d_backprop->filter()))
    return false;

  auto data_layout = plier::tf::as_data_layout(conv2d_backprop->data_layout());

  // Nodes to replace
  auto feature_enc = graph->nodes()->create<loco::FeatureEncode>();
  auto filter_enc = graph->nodes()->create<loco::FilterEncode>();
  auto tr_conv2d = graph->nodes()->create<loco::TransposedConv2D>();
  auto feature_dec = graph->nodes()->create<loco::FeatureDecode>();

  set_feature_enc(feature_enc, data_layout);
  set_filter_enc(filter_enc);
  set_feature_dec(feature_dec, data_layout);

  // Attributes for new TransposedConv2D
  loco::Stride<2> stride;
  loco::Padding2D pad;

  // Get attributes
  {
    if (!stride_2d_from_4d(stride, conv2d_backprop->strides(), data_layout))
      throw oops::UserExn("Unsupported strides", conv2d_backprop->name());

    Padding2DInference infer_pad(conv2d_backprop);

    if (!set_plane_shape(infer_pad.input(), conv2d_backprop->out_backprop(), data_layout))
      throw oops::UserExn("Unsupported out_backprop data_format", conv2d_backprop->name());
    if (!set_plane_shape(infer_pad.output(), conv2d_backprop, data_layout))
      throw oops::UserExn("Unsupported data_format", conv2d_backprop->name());
    if (!set_window(infer_pad.window(), conv2d_backprop->filter()))
      throw oops::UserExn("Unsupported filter shape", conv2d_backprop->name());
    infer_pad.stride() = stride;
    infer_pad.padding() = conv2d_backprop->padding();

    // Run padding infer_pad
    pad = infer_pad();
  }

  // Set attributes
  tr_conv2d->pad()->top(pad.top());
  tr_conv2d->pad()->bottom(pad.bottom());
  tr_conv2d->pad()->left(pad.left());
  tr_conv2d->pad()->right(pad.right());

  tr_conv2d->stride()->vertical(stride.vertical());
  tr_conv2d->stride()->horizontal(stride.horizontal());

  // Update graph
  auto input_node = conv2d_backprop->out_backprop();
  auto filter_node = conv2d_backprop->filter();

  // Update connections
  feature_enc->input(input_node);
  filter_enc->input(filter_node);
  tr_conv2d->ifm(feature_enc);
  tr_conv2d->ker(filter_enc);
  feature_dec->input(tr_conv2d);

  // Replace old conv2d_backprop
  replace(conv2d_backprop).with(feature_dec);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool Conv2DBackpropInputCanonicalizer::transform(TFConv2DBackpropInput *node) const
{
  return canonicalize_conv2d_backprop_input(node->graph(), node);
}

} // namespace tf
} // namespace moco
