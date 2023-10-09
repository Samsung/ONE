/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QImplant.h"

#include <loco.h>
#include <luci/IR/CircleNode.h>

#include <npy.hpp>
#include <json.h>
#include <fstream>
#include <unordered_map>
#include <luci/Profile/CircleNodeOrigin.h>

using namespace q_implant;

#define THROW_UNLESS(cond) \
  if (not(cond))           \
    throw std::runtime_error{#cond};

namespace
{

// Return directory path of given file path
// TODO Find a platform-independent way to do this
std::string directory_path(const std::string &file_path)
{
  const auto pos = file_path.find_last_of("/");
  if (std::string::npos == pos)
    return "";

  return file_path.substr(0, pos);
}

loco::DataType str_to_dtype(const std::string &str)
{
  auto lower_case_str = str;
  std::transform(lower_case_str.begin(), lower_case_str.end(), lower_case_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_case_str.compare("int8") == 0)
    return loco::DataType::S8;
  if (lower_case_str.compare("uint8") == 0)
    return loco::DataType::U8;
  if (lower_case_str.compare("int16") == 0)
    return loco::DataType::S16;
  if (lower_case_str.compare("int32") == 0)
    return loco::DataType::S32;
  if (lower_case_str.compare("int64") == 0)
    return loco::DataType::S64;

  throw std::runtime_error("Invalid dtype detected. " + str);
}

// Throw an exception if tensor has any invalid field.
void verify_tensor(const Json::Value &tensor)
{
  THROW_UNLESS(tensor.isMember("scale"));
  THROW_UNLESS(tensor["scale"].isString());
  THROW_UNLESS(tensor.isMember("zerop"));
  THROW_UNLESS(tensor["zerop"].isString());
  THROW_UNLESS(tensor.isMember("quantized_dimension"));
  THROW_UNLESS(tensor["quantized_dimension"].isUInt());
  THROW_UNLESS(tensor.isMember("dtype"));
  THROW_UNLESS(tensor["dtype"].isString());

  if (tensor.isMember("value"))
  {
    THROW_UNLESS(tensor["value"].isString());
  }
}

Json::Value load_json(const std::string &path)
{
  Json::Value root;
  std::ifstream ifs(path);

  // Failed to open cfg file
  if (not ifs.is_open())
    throw std::runtime_error("Cannot open config file. " + path);

  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;

  // Failed to parse
  if (not parseFromStream(builder, ifs, &root, &errs))
    throw std::runtime_error("Cannot parse config file (json format). " + errs);

  return root;
}

void set_dtype(luci::CircleNode *node, loco::DataType dtype) { node->dtype(dtype); }

void copy_dtype(const luci::CircleNode *src, luci::CircleNode *dest) { dest->dtype(src->dtype()); }

void set_scale(luci::CircleNode *node, const std::string &scale_path)
{
  assert(node);               // FIX CALLER UNLESS
  assert(node->quantparam()); // FIX CALLER UNLESS

  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<float> scale;
  npy::LoadArrayFromNumpy(scale_path, shape, fortran_order, scale);

  THROW_UNLESS(shape.size() == 1);
  THROW_UNLESS(fortran_order == false);

  node->quantparam()->scale = scale;
}

void set_zerop(luci::CircleNode *node, const std::string &zerop_path)
{
  assert(node);               // FIX CALLER UNLESS
  assert(node->quantparam()); // FIX CALLER UNLESS

  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<int64_t> zerop;
  npy::LoadArrayFromNumpy(zerop_path, shape, fortran_order, zerop);

  THROW_UNLESS(shape.size() == 1);
  THROW_UNLESS(fortran_order == false);

  node->quantparam()->zerop = zerop;
}

void set_quantized_dimension(luci::CircleNode *node, const uint32_t quantized_dimension)
{
  assert(node);               // FIX CALLER UNLESS
  assert(node->quantparam()); // FIX CALLER UNLESS

  node->quantparam()->quantized_dimension = quantized_dimension;
}

template <loco::DataType DT> void set_value(luci::CircleConst *node, const std::string &value_path)
{
  assert(node);                // FIX CALLER UNLESS
  assert(node->dtype() == DT); // FIX CALLER UNLESS

  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<typename loco::DataTypeImpl<DT>::Type> values;
  npy::LoadArrayFromNumpy(value_path, shape, fortran_order, values);

  THROW_UNLESS(shape.size() == node->rank());
  THROW_UNLESS(fortran_order == false);

  uint32_t value_size = 1;
  for (uint32_t i = 0; i < node->rank(); i++)
  {
    THROW_UNLESS(node->dim(i).known());
    THROW_UNLESS(node->dim(i).value() == shape[i]);

    value_size *= node->dim(i).value();
  }

  node->size<DT>(value_size);
  for (uint32_t i = 0; i < value_size; i++)
  {
    node->at<DT>(i) = values.at(i);
  }
}

void set_value(luci::CircleConst *const_node, const std::string &value_path, loco::DataType dtype)
{
  assert(const_node); // FIX CALLER UNLESS

  switch (dtype)
  {
    case loco::DataType::S8:
      set_value<loco::DataType::S8>(const_node, value_path);
      break;
    case loco::DataType::U8:
      set_value<loco::DataType::U8>(const_node, value_path);
      break;
    case loco::DataType::S16:
      set_value<loco::DataType::S16>(const_node, value_path);
      break;
    case loco::DataType::S32:
      set_value<loco::DataType::S32>(const_node, value_path);
      break;
    case loco::DataType::S64:
      set_value<loco::DataType::S64>(const_node, value_path);
      break;
    default:
      throw std::runtime_error("Invalid value dtype detected. ");
  }
}

template <loco::DataType DT> void apply_qparam(luci::CircleConst *node, const luci::CircleQuantParam *qparam)
{
  std::vector<typename loco::DataTypeImpl<DT>::Type> values;

  uint32_t quantized_dimension = qparam->quantized_dimension;
  auto zerop = qparam->zerop;
  auto scale = qparam->scale;
  auto mines = qparam->min;
  auto maxes = qparam->max;

  uint32_t channel_size = 1;
  uint32_t value_size = 1;

  if (node->rank() != 0)
  {
    for (uint32_t i = 0; i < quantized_dimension; i++)
    {
      THROW_UNLESS(node->dim(i).known());
      channel_size *= node->dim(i).value();
    }

    for (uint32_t i = quantized_dimension; i < node->rank(); i++)
    {
      THROW_UNLESS(node->dim(i).known());
      value_size *= node->dim(i).value();
    }
  }

  for (uint32_t c = 0; c < channel_size; ++c)
  {
    auto z = zerop.at(c);
    auto s = scale.at(c);
    auto offset = c * value_size;

    float min = std::numeric_limits<float>::lowest();
    float max = std::numeric_limits<float>::max();

    if (mines.size() > c)
      min = mines.at(c);

    if (maxes.size() > c)
      max = maxes.at(c);

    for (uint32_t v = 0; v < value_size; ++v)
    {
      float data = node->at<loco::DataType::FLOAT32>(offset + v);

      data = data < min ? min : data;
      data = data > max ? max : data;

      data = (data - min) / s;
      values.emplace_back(static_cast<typename loco::DataTypeImpl<DT>::Type>(std::round(data)) + z);
    }
  }

  uint32_t total_size = channel_size * value_size;

  node->dtype(DT);
  node->size<DT>(total_size);
  for (uint32_t i = 0; i < total_size; i++)
    node->at<DT>(i) = values.at(i);

  auto copy_qparam = std::make_unique<luci::CircleQuantParam>();
  copy_qparam->scale = scale;
  copy_qparam->zerop = zerop;
  copy_qparam->quantized_dimension = quantized_dimension;

  auto circle_node = loco::must_cast<luci::CircleNode *>(node);
  circle_node->quantparam(std::move(copy_qparam));
}

void apply_qparam(luci::CircleConst *const_node, const luci::CircleQuantParam *qparam, loco::DataType dtype)
{
  assert(const_node->dtype() == loco::DataType::FLOAT32);

  switch (dtype)
  {
    case loco::DataType::S8:
      apply_qparam<loco::DataType::S8>(const_node, qparam);
      break;
    case loco::DataType::U8:
      apply_qparam<loco::DataType::U8>(const_node, qparam);
      break;
    case loco::DataType::S16:
      apply_qparam<loco::DataType::S16>(const_node, qparam);
      break;
    case loco::DataType::S32:
      apply_qparam<loco::DataType::S32>(const_node, qparam);
      break;
    case loco::DataType::S64:
      apply_qparam<loco::DataType::S64>(const_node, qparam);
      break;
    default:
      throw std::runtime_error("Invalid value dtype detected. ");
  }
}

bool recalculate_min_max_require(luci::CircleQuantParam *qparam)
{
  return qparam->max.empty() || qparam->min.empty();
}

template <loco::DataType DT> void recalculate_min_max(luci::CircleQuantParam *qparam)
{
  auto& zero_points = qparam->zerop;
  auto& scales = qparam->scale;

  assert(zero_points.size() == scales.size());
  uint32_t length = zero_points.size();

  assert(recalculate_min_max_require(qparam));
  auto& mins = qparam->min;
  auto& maxes = qparam->max;

  int64_t lower_bound = std::numeric_limits<typename loco::DataTypeImpl<DT>::Type>::lowest();
  int64_t upper_bound = std::numeric_limits<typename loco::DataTypeImpl<DT>::Type>::max();

  for (uint32_t i = 0; i < length; ++i)
  {
    auto zerop = zero_points.at(i);
    auto scale = scales.at(i);

    int64_t min = lower_bound - zerop;
    int64_t max = upper_bound - zerop;

    if(std::is_same<typename loco::DataTypeImpl<DT>::Type, int64_t>())
    {
      if(min > 0)
        min = lower_bound;

      if(max < 0)
        max = upper_bound;
    }

    float calculated_min = static_cast<float >(min) * scale;
    float calculated_max = static_cast<float >(max) * scale;

    mins.emplace_back(calculated_min);
    maxes.emplace_back(calculated_max);
  }
}

void recalculate_min_max(luci::CircleQuantParam *qparam, loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::S8:
      recalculate_min_max<loco::DataType::S8>(qparam);
      break;
    case loco::DataType::U8:
      recalculate_min_max<loco::DataType::U8>(qparam);
      break;
    case loco::DataType::S16:
      recalculate_min_max<loco::DataType::S16>(qparam);
      break;
    case loco::DataType::S32:
      recalculate_min_max<loco::DataType::S32>(qparam);
      break;
    case loco::DataType::S64:
      recalculate_min_max<loco::DataType::S64>(qparam);
      break;
    default:
      throw std::runtime_error("Invalid value dtype detected. ");
  }
}

template <loco::DataType DT> std::unique_ptr<luci::CircleQuantParam> extend_qparam(const luci::CircleQuantParam *x, const luci::CircleQuantParam *y, void (*f)(double &, double &, double &, double &, double &, double &))
{
  assert(x->quantized_dimension == y->quantized_dimension);
  auto& x_zero_points = x->zerop;
  auto& x_scales = x->scale;
  auto& y_zero_points = y->zerop;
  auto& y_scales = y->scale;
  assert(x_scales.size() == x_zero_points.size());
  assert(y_scales.size() == y_zero_points.size());
  assert(x_scales.size() == y_scales.size());

  auto& x_mins = x->min;
  auto& x_maxes = x->max;
  assert(x_mins.size() == x_maxes.size());
  auto& y_mins = y->min;
  auto& y_maxes = y->max;
  assert(y_mins.size() == y_maxes.size());
  assert(x_mins.size() == x_scales.size());

  uint32_t length = x_scales.size();
  int64_t lower_bound = std::numeric_limits<typename loco::DataTypeImpl<DT>::Type>::lowest();
  int64_t upper_bound = std::numeric_limits<typename loco::DataTypeImpl<DT>::Type>::max();
  auto lower_bound_double = static_cast<double >(lower_bound);
  auto upper_bound_double = static_cast<double >(upper_bound);

  auto ret_qparam = std::make_unique<luci::CircleQuantParam>();

  for(uint32_t i = 0; i < length; ++i)
  {
    auto x_scale = x_scales.at(i);
    auto x_scale_inv = 1 / x_scale;
    auto x_zerop = x_zero_points.at(i);
    auto y_scale = y_scales.at(i);
    auto y_scale_inv = 1 / y_scale;
    auto y_zerop = y_zero_points.at(i);
    int64_t x_min = static_cast<int64_t >(x_mins.at(i) * x_scale_inv) - x_zerop;
    int64_t x_max = static_cast<int64_t >(x_maxes.at(i) * x_scale_inv) - x_zerop;
    int64_t y_min = static_cast<int64_t >(y_mins.at(i) * y_scale_inv) - y_zerop;
    int64_t y_max = static_cast<int64_t >(y_maxes.at(i) * y_scale_inv) - y_zerop;

    auto x_min_double = static_cast<double >(x_min);
    auto x_max_double = static_cast<double >(x_max);
    auto y_min_double = static_cast<double >(y_min);
    auto y_max_double = static_cast<double >(y_max);
    double min_ret;
    double max_ret;
    f(x_min_double, x_max_double, y_min_double, y_max_double, min_ret, max_ret);

    if (min_ret < lower_bound_double)
      min_ret = lower_bound_double;

    if(max_ret > upper_bound_double)
      max_ret = upper_bound_double;

    double scale = (max_ret - min_ret) / (upper_bound_double - lower_bound_double);
    double zerop_double;

    if(scale == 0)
      zerop_double = (upper_bound_double + lower_bound_double)/ 2;
    else
      zerop_double = min_ret - lower_bound_double / scale;

    ret_qparam->scale.emplace_back(static_cast<float >(scale));
    ret_qparam->zerop.emplace_back(static_cast<int64_t >(zerop_double));
    ret_qparam->min.emplace_back(static_cast<float >(min_ret));
    ret_qparam->max.emplace_back(static_cast<float >(max_ret));
  }

  return ret_qparam;
}

std::unique_ptr<luci::CircleQuantParam> extend_qparam(luci::CircleQuantParam *x, luci::CircleQuantParam *y, loco::DataType dtype, void (*f)(double &, double &, double &, double &, double &, double &))
{
  switch (dtype)
  {
    case loco::DataType::S8:
      return extend_qparam<loco::DataType::S8>(x, y, f);
    case loco::DataType::U8:
      return extend_qparam<loco::DataType::U8>(x, y, f);
    case loco::DataType::S16:
      return extend_qparam<loco::DataType::S16>(x, y, f);
    case loco::DataType::S32:
      return extend_qparam<loco::DataType::S32>(x, y, f);
    case loco::DataType::S64:
      return extend_qparam<loco::DataType::S64>(x, y, f);
    default:
      throw std::runtime_error("Invalid value dtype detected. ");
  }
}


} // namespace

void QImplant::write(loco::Graph *g)
{
  const auto root = load_json(_path);
  const auto dir_path = directory_path(_path);

  std::unordered_map<std::string, luci::CircleNode *> name_to_node;
  for (auto node : loco::all_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
    {
      continue;
    }

    name_to_node[circle_node->name()] = circle_node;
  }

  THROW_UNLESS(root.isObject());

  for (const auto tensor_name : root.getMemberNames())
  {
    const auto tensor = root[tensor_name];

    verify_tensor(tensor);

    const auto scale_path = dir_path + '/' + tensor["scale"].asString();
    const auto zerop_path = dir_path + '/' + tensor["zerop"].asString();
    const auto quantized_dimension = tensor["quantized_dimension"].asUInt();
    const auto dtype = str_to_dtype(tensor["dtype"].asString());

    auto node = name_to_node.at(tensor_name);

    // Node must be fp32
    THROW_UNLESS(node->dtype() == loco::DataType::FLOAT32);

    node->quantparam(std::make_unique<luci::CircleQuantParam>());

    set_dtype(node, dtype);
    set_scale(node, scale_path);
    set_zerop(node, zerop_path);
    set_quantized_dimension(node, quantized_dimension);

    if (tensor.isMember("value"))
    {
      auto const_node = loco::must_cast<luci::CircleConst *>(node);
      const auto value_path = dir_path + '/' + tensor["value"].asString();

      set_value(const_node, value_path, dtype);
    }
  }

  forward_qparam(g);

  // Update output nodes
  auto graph_outputs = g->outputs();
  assert(graph_outputs); // FIX_CALLER_UNLESS
  for (auto node : loco::output_nodes(g))
  {
    auto out_node = loco::must_cast<luci::CircleOutput *>(node);
    auto from_node = loco::must_cast<luci::CircleNode *>(out_node->from());

    THROW_UNLESS(from_node->quantparam());

    out_node->quantparam(std::make_unique<luci::CircleQuantParam>());
    out_node->quantparam()->scale = from_node->quantparam()->scale;
    out_node->quantparam()->zerop = from_node->quantparam()->zerop;
    out_node->quantparam()->quantized_dimension = from_node->quantparam()->quantized_dimension;
    out_node->dtype(from_node->dtype());

    auto graph_output = graph_outputs->at(out_node->index());
    graph_output->dtype(out_node->dtype());
  }

  // Verify quantized model
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    // Throw an exception if dtype is not float32
    // TODO Operator-level verification (ex: using QuantizedModelVerifier)
    THROW_UNLESS(circle_node->dtype() != loco::DataType::FLOAT32);
  }

  for (auto node : loco::input_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    auto quantize = node->graph()->nodes()->create<luci::CircleQuantize>();
    quantize->name(circle_node->name() + "_Quantize");
    quantize->dtype(circle_node->dtype());
    quantize->rank(circle_node->rank());
    for (uint32_t i = 0; i < circle_node->rank(); ++i)
      quantize->dim(i).set(circle_node->dim(i).value());

    quantize->shape_status(luci::ShapeStatus::VALID);

    copy_quantparam(circle_node, quantize);
    circle_node->quantparam(nullptr);
    circle_node->dtype(loco::DataType::FLOAT32);

    loco::replace(circle_node).with(quantize);
    quantize->input(circle_node);
    luci::add_origin(quantize, luci::get_origin(circle_node));
  }

    // fail on loco::DataType visit(const luci::CircleOutput *node) on CircleTypeInferenceRule.cpp
    // output_dtype == luci::dtype_get(node->from())
//  for (auto node : loco::output_nodes(g))
//  {
//    auto output_node = loco::must_cast<luci::CircleNode *>(node);
//    output_node->dtype(loco::DataType::FLOAT32);
//    for (auto pred : loco::preds(node))
//    {
//      auto circle_node = loco::must_cast<luci::CircleNode *>(pred);
//      auto dequantize = pred->graph()->nodes()->create<luci::CircleDequantize>();
//      dequantize->name(output_node->name() + "_DeQuantize");
//      dequantize->dtype(circle_node->dtype());
//      dequantize->rank(circle_node->rank());
//
//      for (uint32_t i = 0; i < circle_node->rank(); ++i)
//        dequantize->dim(i).set(circle_node->dim(i).value());
//
//      dequantize->shape_status(luci::ShapeStatus::VALID);
//      copy_quantparam(circle_node, dequantize);
//      loco::replace(circle_node).with(dequantize);
//      dequantize->input(circle_node);
//      luci::add_origin(dequantize, luci::get_origin(circle_node));
//    }
//  }
}

void QImplant::forward_qparam(loco::Graph *g)
{
  /*
   * TODO: add comment about how to add to the set
   *
   * If the operator doesn't change input tensor value,
   * (Operator don't change input quantization parameter to output tensor)
   * the operator's quantization parameter can be forwarded
   */
  std::set<luci::CircleOpcode> forwardable_opcode;
  forwardable_opcode.emplace(luci::CircleOpcode::RESHAPE);
  forwardable_opcode.emplace(luci::CircleOpcode::SPLIT);
  forwardable_opcode.emplace(luci::CircleOpcode::CIRCLESPLITOUT);
  forwardable_opcode.emplace(luci::CircleOpcode::TRANSPOSE);
  // TODO add more Ops

  // maybe not accurate, just for experimental purpose
  forwardable_opcode.emplace(luci::CircleOpcode::PAD);
  forwardable_opcode.emplace(luci::CircleOpcode::MEAN);
  forwardable_opcode.emplace(luci::CircleOpcode::PADV2);
  forwardable_opcode.emplace(luci::CircleOpcode::MAX_POOL_2D);


  auto forwardable = [&forwardable_opcode](luci::CircleOpcode opcode) {
    return forwardable_opcode.find(opcode) != forwardable_opcode.end();
  };


  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    auto quantparam = circle_node->quantparam();

    if (circle_node->opcode() == luci::CircleOpcode::PADV2){
      auto pad_v2 = reinterpret_cast<luci::CirclePadV2 *>(circle_node);
      auto constant_values_node = loco::must_cast<luci::CircleConst *>(pad_v2->constant_values());
      if (constant_values_node->quantparam() == nullptr && constant_values_node->dtype() == loco::DataType::FLOAT32){
        apply_qparam(constant_values_node, quantparam, circle_node->dtype());
      }
    }

    if (quantparam == nullptr){
      if(circle_node->opcode() == luci::CircleOpcode::ADD)
      {
        auto add = reinterpret_cast<luci::CircleAdd *>(circle_node);
        auto x_node = loco::must_cast<luci::CircleNode *>(add->x());
        auto y_node = loco::must_cast<luci::CircleNode *>(add->y());
        auto x_qparam = x_node->quantparam();
        auto y_qparam = y_node->quantparam();

        if(x_qparam == nullptr || y_qparam == nullptr)
        {
          // no op
        }
        else if (x_qparam->scale == y_qparam->scale && x_qparam->zerop == y_qparam->zerop)
        {
          copy_quantparam(x_node, add);
          copy_dtype(x_node, add);
        }
        else if(x_node->dtype() == y_node->dtype())
        {
          if(recalculate_min_max_require(x_qparam))
          {
            recalculate_min_max(x_qparam, x_node->dtype());
          }

          if(recalculate_min_max_require(y_qparam))
          {
            recalculate_min_max(y_qparam, y_node->dtype());
          }

          auto z_qparam = extend_qparam(x_qparam, y_qparam, x_node->dtype(),
                                        [](double &x_min, double &x_max, double &y_min, double &y_max, double &ret_min, double &ret_max){
            ret_min = std::min(x_min, y_min);
            ret_min = std::min(ret_min, x_min + y_min);
            ret_max = std::max(x_max, y_max);
            ret_max = std::max(ret_max, x_max + y_max);
          }
          );

          add->dtype(x_node->dtype());
          add->quantparam(std::move(z_qparam));
        }
      }
      else
        continue;
    }

    for (auto successor : loco::succs(node))
    {
      auto successor_node = loco::must_cast<luci::CircleNode *>(successor);

      if (successor_node->quantparam() == nullptr)
      {
        if (!forwardable(successor_node->opcode()))
          continue;
        copy_quantparam(circle_node, successor_node);
        copy_dtype(circle_node, successor_node);
      }
    }
  }
}

#undef THROW_UNLESS
