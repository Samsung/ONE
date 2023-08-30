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
}

#undef THROW_UNLESS
