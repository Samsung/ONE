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

#include "luci/Pass/ResolveCustomOpMaxPoolWithArgmaxPass.h"

#include "flatbuffers/flexbuffers.h"
#include <loco/IR/DataTypeTraits.h>

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <loco.h>
#include <oops/InternalExn.h>

namespace
{

template <typename T> std::vector<T> to_vector(const flexbuffers::TypedVector &typed_vec)
{
  std::vector<T> answer(typed_vec.size());

  for (uint32_t i = 0; i < answer.size(); ++i)
  {
    answer[i] = typed_vec[i].As<T>();
  }

  return answer;
}

luci::Padding string_to_padding(const std::string &pad_str)
{
  if (pad_str == "VALID")
    return luci::Padding::VALID;
  if (pad_str == "SAME")
    return luci::Padding::SAME;

  return luci::Padding::UNDEFINED;
}

template <typename NodeT> void set_stride(NodeT *node, const luci::Stride &stride)
{
  node->stride()->h(stride.h());
  node->stride()->w(stride.w());
}

template <typename NodeT> void set_filter(NodeT *node, const luci::Filter &filter)
{
  node->filter()->h(filter.h());
  node->filter()->w(filter.w());
}

void init_name_and_origin(luci::CircleNode *node, const std::string &name,
                          const std::shared_ptr<luci::CircleNodeOrigin> &origin)
{
  node->name(name);
  luci::add_origin(node, origin);
}

template <typename NodeT> NodeT *none_act_func(NodeT *node)
{
  node->fusedActivationFunction(luci::FusedActFunc::NONE);
  return node;
}

luci::CircleCast *create_cast(luci::CircleNode *input, loco::DataType in_type,
                              loco::DataType out_type)
{
  auto cast = input->graph()->nodes()->create<luci::CircleCast>();

  cast->in_data_type(in_type);
  cast->out_data_type(out_type);
  cast->dtype(out_type);

  cast->x(input);

  return cast;
}

template <loco::DataType DT> void fill_conv_weights(luci::CircleConst *weights)
{
  assert(weights->rank() == 4);

  auto const kn = weights->dim(0).value();
  auto const kh = weights->dim(1).value();
  auto const kw = weights->dim(2).value();

  auto elements_size = kn * kh * kw * 1;
  weights->size<DT>(elements_size);

  for (uint32_t b = 0; b < kn; ++b)
  {
    for (uint32_t y = 0; y < kh; ++y)
    {
      for (uint32_t x = 0; x < kw; ++x)
      {
        auto const idx = (b * kh + y) * kw + x;
        weights->at<DT>(idx) = (y * kw + x == b) ? 1 : 0;
      }
    }
  }
}

luci::CircleConst *create_conv_filter(luci::CircleConv2D *conv, const uint32_t kh,
                                      const uint32_t kw, const uint32_t kn)
{
  auto weights = conv->graph()->nodes()->create<luci::CircleConst>();

  weights->dtype(loco::DataType::FLOAT32);

  weights->rank(4);
  weights->dim(0).set(kn);
  weights->dim(1).set(kh);
  weights->dim(2).set(kw);
  weights->dim(3).set(1);
  weights->shape_status(luci::ShapeStatus::VALID);

  fill_conv_weights<loco::DataType::FLOAT32>(weights);

  return weights;
}

template <loco::DataType DT> void fill_zero_bias(luci::CircleConst *bias)
{
  assert(bias->rank() == 1);
  auto const depth = bias->dim(0).value();

  bias->size<DT>(depth);

  for (uint32_t i = 0; i < depth; ++i)
  {
    bias->at<DT>(i) = 0;
  }
}

luci::CircleConst *create_zero_bias(loco::Graph *graph, uint32_t depth)
{
  auto bias = graph->nodes()->create<luci::CircleConst>();

  bias->dtype(loco::DataType::FLOAT32);

  bias->rank(1);
  bias->dim(0).set(depth);

  fill_zero_bias<loco::DataType::FLOAT32>(bias);

  return bias;
}

template <loco::DataType DT, typename Numeric>
luci::CircleConst *create_scalar(loco::Graph *graph, Numeric value)
{
  auto scalar = graph->nodes()->create<luci::CircleConst>();

  scalar->dtype(DT);

  scalar->rank(0);
  scalar->size<DT>(1);
  scalar->shape_status(luci::ShapeStatus::VALID);

  scalar->scalar<DT>() = value;

  return scalar;
}

luci::CircleConst *create_shape_tensor(loco::Graph *graph, const std::vector<uint32_t> &dims_vec)
{
  auto shape = graph->nodes()->create<luci::CircleConst>();

  shape->dtype(loco::DataType::S32);

  shape->rank(1);
  shape->dim(0).set(dims_vec.size());
  shape->shape_status(luci::ShapeStatus::VALID);

  shape->size<loco::DataType::S32>(dims_vec.size());

  for (uint32_t i = 0; i < dims_vec.size(); ++i)
  {
    shape->at<loco::DataType::S32>(i) = dims_vec[i];
  }

  return shape;
}

template <loco::DataType DT>
void fill_coords_addition(luci::Padding padding, const luci::Stride &stride,
                          const luci::Filter &filter, uint32_t input_width,
                          luci::CircleConst *cords)
{
  assert(cords->rank() == 4);

  auto const output_height = static_cast<int32_t>(cords->dim(1).value());
  auto const output_width = static_cast<int32_t>(cords->dim(2).value());
  {
    auto const element_counts = 1 * output_height * output_width * 1;
    cords->size<DT>(element_counts);
  }

  assert(padding != luci::Padding::UNDEFINED);

  // For VALID padding:
  int32_t start_y = 0;
  int32_t start_x = 0;

  // For SAME padding:
  if (padding == luci::Padding::SAME)
  {
    start_y -= static_cast<int32_t>(filter.h() - 1) / 2;
    start_x -= static_cast<int32_t>(filter.w() - 1) / 2;
  }

  auto const step_y = static_cast<int32_t>(stride.h());
  auto const step_x = static_cast<int32_t>(stride.w());

  for (int32_t y_o = 0, y_i = start_y; y_o < output_height; ++y_o, y_i += step_y)
  {
    for (int32_t x_o = 0, x_i = start_y; x_o < output_width; ++x_o, x_i += step_x)
    {
      auto const output_idx = y_o * output_width + x_o;
      auto const input_idx = y_i * input_width + x_i;

      cords->at<DT>(output_idx) = input_idx;
    }
  }
}

luci::CircleConst *create_coords_addition(loco::Graph *graph, luci::Padding padding,
                                          const luci::Stride &stride, const luci::Filter &filter,
                                          uint32_t, uint32_t input_width, uint32_t output_height,
                                          uint32_t output_width)
{
  auto cords = graph->nodes()->create<luci::CircleConst>();

  cords->dtype(loco::DataType::FLOAT32);

  cords->rank(4);
  cords->dim(0).set(1);
  cords->dim(1).set(output_height);
  cords->dim(2).set(output_width);
  cords->dim(3).set(1);

  fill_coords_addition<loco::DataType::FLOAT32>(padding, stride, filter, input_width, cords);

  return cords;
}

luci::CircleNode *get_custom_output(const luci::CircleCustom *cop, int32_t idx)
{
  auto const outputs = loco::succs(cop);
  assert(outputs.size() == 2);

  auto output = loco::must_cast<luci::CircleCustomOut *>(*outputs.begin());
  if (output->index() != idx)
  {
    output = loco::must_cast<luci::CircleCustomOut *>(*outputs.rbegin());
  }

  return output;
}

luci::CircleNode *max_pool_branch(luci::Padding padding, const luci::Stride &stride,
                                  const luci::Filter filter, luci::CircleCustom *cop)
{
  auto graph = cop->graph();
  auto input = cop->inputs(0);

  auto origin = luci::get_origin(cop);
  auto name = cop->name() + "/Argmax";

  // Create MaxPool
  auto maxpool = none_act_func(graph->nodes()->create<luci::CircleMaxPool2D>());
  {
    init_name_and_origin(maxpool, name + "/MaxPool2D", origin);

    set_stride(maxpool, stride);
    set_filter(maxpool, filter);
    maxpool->padding(padding);

    maxpool->value(input);
  }

  return maxpool;
}

luci::CircleNode *window_flattened_coord(const std::string &name, luci::Padding padding,
                                         const luci::Stride &stride, const luci::Filter filter,
                                         uint32_t output_height, uint32_t output_width,
                                         luci::CircleNode *input)
{
  auto const graph = input->graph();
  auto const origin = luci::get_origin(input);

  auto const depth_dimension = 3;

  // Create Conv2D
  auto conv = none_act_func(graph->nodes()->create<luci::CircleConv2D>());
  {
    init_name_and_origin(conv, name + "/Conv2D", origin);

    // Padding, Stride and kernel size equal to MaxPool's
    set_stride(conv, stride);
    conv->padding(padding);

    // depth of kernel is equal to square size
    auto const kh = filter.h();
    auto const kw = filter.w();
    auto const kd = kh * kw;

    // use zero bias
    auto bias = create_zero_bias(graph, kd);
    init_name_and_origin(bias, conv->name() + "/Bias", origin);

    // create filter
    // TODO make shared
    auto weights = create_conv_filter(conv, kh, kw, kd);
    init_name_and_origin(weights, conv->name() + "/Weights", origin);

    conv->bias(bias);
    conv->filter(weights);
    conv->input(input);
  }

  // Create ArgMax
  auto argmax = graph->nodes()->create<luci::CircleArgMax>();
  {
    init_name_and_origin(argmax, name + "/ArgMax", origin);

    argmax->output_type(loco::DataType::S32);

    // Create argmax_dim
    auto argmax_dim = create_scalar<loco::DataType::S32>(graph, depth_dimension);
    init_name_and_origin(argmax_dim, argmax->name() + "/Dimension", origin);

    argmax->dimension(argmax_dim);
    argmax->input(conv);
  }

  // Create Reshape to 4-rank back, because argmax decrease rank of tensor by 1
  auto reshape = graph->nodes()->create<luci::CircleReshape>();
  {
    init_name_and_origin(reshape, name + "/Reshape", origin);

    auto shape = create_shape_tensor(graph, {1, output_height, output_width, 1});
    init_name_and_origin(shape, reshape->name() + "/Shape", origin);

    reshape->tensor(argmax);
    reshape->shape(shape);
  }

  // Create Cast to use float32 instead int32
  auto argmax_cast = create_cast(reshape, loco::DataType::S32, loco::DataType::FLOAT32);
  init_name_and_origin(argmax_cast, argmax->name() + "/Cast", origin);

  return argmax_cast;
}

luci::CircleNode *window_y_coord(const std::string &name, float filter_width,
                                 luci::CircleNode *flattened)
{
  auto const graph = flattened->graph();
  auto const origin = luci::get_origin(flattened);

  auto div = none_act_func(graph->nodes()->create<luci::CircleMul>());
  {
    init_name_and_origin(div, name + "/Div", origin);

    auto divider = create_scalar<loco::DataType::FLOAT32>(graph, 1.0f / filter_width);
    init_name_and_origin(divider, div->name() + "/Divider", origin);

    div->x(flattened);
    div->y(divider);
  }

  auto floor = graph->nodes()->create<luci::CircleFloor>();
  {
    init_name_and_origin(floor, name + "/Floor", origin);
    floor->x(div);
  }

  return floor;
}

luci::CircleNode *window_x_coord(const std::string &name, float filter_width,
                                 luci::CircleNode *flattened, luci::CircleNode *y_coord)
{
  auto const graph = flattened->graph();
  auto const origin = luci::get_origin(flattened);

  auto mod = none_act_func(graph->nodes()->create<luci::CircleAdd>());
  {
    init_name_and_origin(mod, name + "/Mod", origin);

    auto neg = graph->nodes()->create<luci::CircleNeg>();
    {
      init_name_and_origin(neg, mod->name() + "/Neg", origin);

      auto mul = none_act_func(graph->nodes()->create<luci::CircleMul>());
      {
        init_name_and_origin(mul, neg->name() + "/Neg", origin);

        auto multipler = create_scalar<loco::DataType::FLOAT32>(graph, filter_width);
        init_name_and_origin(multipler, mul->name() + "/Multipler", origin);

        mul->x(y_coord);
        mul->y(multipler);
      }

      neg->x(mul);
    }

    mod->x(flattened);
    mod->y(neg);
  }

  return mod;
}

luci::CircleNode *plane_flattened_coord(const std::string &name, uint32_t input_width,
                                        luci::CircleNode *y_coord, luci::CircleNode *x_coord,
                                        luci::CircleNode *corners)
{
  auto const graph = corners->graph();
  auto const origin = luci::get_origin(corners);

  auto add = none_act_func(graph->nodes()->create<luci::CircleAdd>());
  {
    init_name_and_origin(add, name + "/Add", origin);

    auto addition = none_act_func(graph->nodes()->create<luci::CircleAdd>());
    {
      init_name_and_origin(addition, add->name() + "/Add", origin);

      auto y_addition = none_act_func(graph->nodes()->create<luci::CircleMul>());
      {
        init_name_and_origin(y_addition, addition->name() + "/Mul", origin);

        auto width_scalar = create_scalar<loco::DataType::FLOAT32>(graph, input_width);
        init_name_and_origin(width_scalar, y_addition->name() + "/Const", origin);

        y_addition->x(y_coord);
        y_addition->y(width_scalar);
      }

      addition->x(x_coord);
      addition->y(y_addition);
    }

    add->x(addition);
    add->y(corners);
  }

  return add;
}

luci::CircleNode *volume_flattened_coords(const std::string &name, uint32_t channel,
                                          uint32_t input_depth, luci::CircleNode *plane)
{
  auto const graph = plane->graph();
  auto const origin = luci::get_origin(plane);

  // Create Mul
  auto mul = none_act_func(graph->nodes()->create<luci::CircleMul>());
  {
    init_name_and_origin(mul, name + "/Mul", origin);

    auto depth_scalar = create_scalar<loco::DataType::FLOAT32>(graph, input_depth);
    init_name_and_origin(depth_scalar, mul->name() + "/Const", origin);

    mul->x(plane);
    mul->y(depth_scalar);
  }

  luci::CircleNode *volume = mul;

  // Add channel number to output
  if (channel > 0)
  {
    // Create Add
    auto add_ch = none_act_func(graph->nodes()->create<luci::CircleAdd>());
    init_name_and_origin(add_ch, name + "/Add_Channel", origin);

    auto channel_scalar = create_scalar<loco::DataType::FLOAT32>(graph, channel);
    init_name_and_origin(channel_scalar, add_ch->name() + "/Const", origin);

    add_ch->x(mul);
    add_ch->y(channel_scalar);

    volume = add_ch;
  }

  return volume;
}

luci::CircleNode *argmax_branch(luci::Padding padding, const luci::Stride &stride,
                                const luci::Filter filter, luci::CircleCustom *cop)
{
  auto graph = cop->graph();
  auto input = loco::must_cast<luci::CircleNode *>(cop->inputs(0));
  auto output = get_custom_output(cop, 1);

  auto const depth_dimension = 3;
  auto const input_depth = input->dim(depth_dimension).value();
  auto const input_height = input->dim(1).value();
  auto const input_width = input->dim(2).value();

  assert(output->rank() == 4);
  auto const output_height = output->dim(1).value();
  auto const output_width = output->dim(2).value();

  auto origin = luci::get_origin(cop);
  auto name = cop->name() + "/Argmax";

  // Create Split
  auto split = graph->nodes()->create<luci::CircleSplit>();
  {
    init_name_and_origin(split, name + "/Split", origin);

    // Create split_dim
    auto split_dim = create_scalar<loco::DataType::S32>(graph, depth_dimension);
    init_name_and_origin(split_dim, split->name() + "/Dim", origin);

    split->num_split(int32_t(input_depth));

    split->split_dim(split_dim);
    split->input(input);
  }

  /**
   * Note: we need define idx from input_tensor of maximum element in MaxPool's sliding window.
   * For this we split input tensor by channels, define idx in sliding window and convert this idx
   * to idx from source input_tensor using FloorDiv, Mul and Add operations with constant tensors.
   */
  std::vector<luci::CircleNode *> branch_outputs(input_depth);

  for (uint32_t br_n = 0; br_n < input_depth; ++br_n)
  {
    auto const branch_name = name + "/depth_" + std::to_string(br_n);

    // Create CircleSplitOut
    auto split_out = graph->nodes()->create<luci::CircleSplitOut>();
    init_name_and_origin(split_out, branch_name + "/SplitOut", origin);
    split_out->index(int32_t(br_n));
    split_out->input(split);

    // Define idx of max element in Window:
    auto window_coords = window_flattened_coord(branch_name + "/WindowFlat", padding, stride,
                                                filter, output_height, output_width, split_out);

    auto const window_y = window_y_coord(branch_name + "/WindowY", filter.w(), window_coords);
    auto const window_x =
      window_x_coord(branch_name + "/WindowX", filter.w(), window_coords, window_y);

    // Define idx of max element in Plane
    // This tensor contains coords of left top corners for each window from input tensor
    auto corners = create_coords_addition(graph, padding, stride, filter, input_height, input_width,
                                          output_height, output_width);
    init_name_and_origin(corners, branch_name + "/Const", origin);

    auto plane_coord =
      plane_flattened_coord(branch_name + "/PlaneFlat", input_width, window_y, window_x, corners);

    // Define volume coords as final value
    branch_outputs[br_n] =
      volume_flattened_coords(branch_name + "/VolumeFlat", br_n, input_depth, plane_coord);
  }

  // Create Concatenation
  auto concat = none_act_func(graph->nodes()->create<luci::CircleConcatenation>(input_depth));
  {
    init_name_and_origin(concat, name + "/Concatenation", origin);
    concat->axis(depth_dimension);

    for (int32_t i = 0; i < input_depth; ++i)
    {
      concat->values(i, branch_outputs[i]);
    }
  }

  // Output of argmax_with_maxpool should be S64
  auto output_cast = create_cast(concat, loco::DataType::FLOAT32, loco::DataType::S64);
  init_name_and_origin(output_cast, name + "/Cast", origin);

  return output_cast;
}

bool resolve_max_pool_with_argmax(luci::CircleCustom *cop)
{
#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

  const std::vector<uint8_t> custom_options = cop->custom_options();
  auto map = flexbuffers::GetRoot(custom_options).AsMap();

  // Define params
  // Note: Only `Targmax` equal to DT_INT64 is supported by tflite converter
  // Note: Only `data_format` equal to "NHWC" is supported by tflite converter
  // TODO add support of `include_batch_in_index` param
  auto ksize_param = to_vector<uint32_t>(map["ksize"].AsTypedVector());
  auto strides_param = to_vector<uint32_t>(map["strides"].AsTypedVector());
  auto padding_param = map["padding"].As<std::string>();

  // Batch size and depth of ksize more than 1 is not supported.
  CHECK_OR_FALSE(ksize_param.size() == 4);
  CHECK_OR_FALSE(ksize_param[0] == 1 && ksize_param[3] == 1);

  CHECK_OR_FALSE(strides_param.size() == 4);
  CHECK_OR_FALSE(strides_param[0] == 1 && strides_param[3] == 1);

  // define Padding
  auto padding = string_to_padding(padding_param);

  // define Filter
  luci::Filter filter;
  filter.h(ksize_param[1]);
  filter.w(ksize_param[2]);

  // define Stride
  luci::Stride stride;
  stride.h(strides_param[1]);
  stride.w(strides_param[2]);

  // input node
  auto const input = loco::must_cast<luci::CircleNode *>(cop->inputs(0));
  CHECK_OR_FALSE(input->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(input->rank() == 4);

  // TODO support batch size > 1 and `include_batch_in_index` option
  CHECK_OR_FALSE(input->dim(0).value() == 1);

  // output nodes
  auto const outputs = loco::succs(cop);
  CHECK_OR_FALSE(outputs.size() == 2);
  assert(outputs.size() == cop->numOutputs());

  auto output0 = get_custom_output(cop, 0);
  auto output1 = get_custom_output(cop, 1);

  // From TF documentation: output of maxpool must has same type as input
  assert(output0->dtype() == input->dtype());
  assert(output1->dtype() == loco::DataType::S64);

  // Create MaxPool
  auto maxpool = max_pool_branch(padding, stride, filter, cop);
  auto argmax = argmax_branch(padding, stride, filter, cop);

  // replace old node with new subgraph
  cop->inputs(0, nullptr);
  loco::replace(output0).with(maxpool);
  loco::replace(output1).with(argmax);

  return true;
}

} // namespace

namespace luci
{

bool ResolveCustomOpMaxPoolWithArgmaxPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (cop->custom_code() != "MaxPoolWithArgmax")
      continue;

    if (!resolve_max_pool_with_argmax(cop))
      continue;

    changed = true;
  }

  return changed;
}

} // namespace luci
