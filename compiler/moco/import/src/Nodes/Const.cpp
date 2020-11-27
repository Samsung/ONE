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

#include "moco/Import/Nodes/Const.h"

#include <moco/Names.h>
#include <moco/IR/TFNodes.h>

#include <loco.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace
{

using namespace moco;

void read_value_int8(TFConst *const_node, int num_elements,
                     const tensorflow::TensorProto &input_tensor)
{
  const_node->size<loco::DataType::S8>(num_elements);

  int32_t input_elements = input_tensor.int_val_size();

  if (input_tensor.tensor_content().size() == num_elements * sizeof(int8_t))
  {
    const std::string &str_content = input_tensor.tensor_content();
    const int8_t *s8_ptr = reinterpret_cast<const int8_t *>(str_content.c_str());
    for (int32_t i = 0; i < num_elements; i++)
    {
      const_node->at<loco::DataType::S8>(i) = *(s8_ptr + i);
    }
  }
  else if (0 < input_elements && input_elements <= num_elements)
  {
    for (int32_t i = 0; i < input_elements; i++)
    {
      const_node->at<loco::DataType::S8>(i) = input_tensor.int_val(i);
    }

    for (int32_t i = input_elements; i < num_elements; i++)
    {
      const_node->at<loco::DataType::S8>(i) = input_tensor.int_val(input_elements - 1);
    }
  }
  else
  {
    throw oops::UserExn("Invalid Const values", const_node->name());
  }
}

void read_value_int32(TFConst *const_node, int num_elements,
                      const tensorflow::TensorProto &input_tensor)
{
  const_node->size<loco::DataType::S32>(num_elements);

  int32_t input_elements = input_tensor.int_val_size();

  if (input_tensor.tensor_content().size() == num_elements * sizeof(int32_t))
  {
    const std::string &str_content = input_tensor.tensor_content();
    const int32_t *s32_ptr = reinterpret_cast<const int32_t *>(str_content.c_str());
    for (int32_t i = 0; i < num_elements; i++)
    {
      const_node->at<loco::DataType::S32>(i) = *(s32_ptr + i);
    }
  }
  else if (0 < input_elements && input_elements <= num_elements)
  {
    for (int32_t i = 0; i < input_elements; i++)
    {
      const_node->at<loco::DataType::S32>(i) = input_tensor.int_val(i);
    }

    for (int32_t i = input_elements; i < num_elements; i++)
    {
      const_node->at<loco::DataType::S32>(i) = input_tensor.int_val(input_elements - 1);
    }
  }
  else
  {
    throw oops::UserExn("Invalid Const values", const_node->name());
  }
}

void read_value_float32(TFConst *const_node, int num_elements,
                        const tensorflow::TensorProto &input_tensor)
{
  const_node->size<loco::DataType::FLOAT32>(num_elements);

  int32_t input_elements = input_tensor.float_val_size();

  if (input_tensor.tensor_content().size() == num_elements * sizeof(float))
  {
    const std::string &str_content = input_tensor.tensor_content();
    const float *float_ptr = reinterpret_cast<const float *>(str_content.c_str());
    for (int32_t i = 0; i < num_elements; i++)
    {
      const_node->at<loco::DataType::FLOAT32>(i) = *(float_ptr + i);
    }
  }
  else if (0 < input_elements && input_elements <= num_elements)
  {
    for (int32_t i = 0; i < input_elements; i++)
    {
      const_node->at<loco::DataType::FLOAT32>(i) = input_tensor.float_val(i);
    }

    for (int32_t i = input_elements; i < num_elements; i++)
    {
      const_node->at<loco::DataType::FLOAT32>(i) = input_tensor.float_val(input_elements - 1);
    }
  }
  else
  {
    throw oops::UserExn("Invalid Const values", const_node->name());
  }
}

} // namespace

namespace moco
{

bool ConstGraphBuilder::validate(const tensorflow::NodeDef &node) const
{
  if (!plier::tf::has_attrs(node, {"dtype", "value"}))
    return false;

  const auto &input_tensor = plier::tf::get_tensor_attr(node, "value");
  const auto &input_shape = input_tensor.tensor_shape();
  const auto &input_dims = input_shape.dim();

  if (!(input_shape.dim_size() <= 6))
    return false;

  for (auto &d : input_dims)
  {
    if (d.size() > std::numeric_limits<int>::max())
      throw oops::UserExn("Const Shape element overflows", node.name());

    if (d.size() < 0)
      throw oops::UserExn("Unknown dim size", node.name());
  }

  auto dtype = plier::tf::as_loco_datatype(plier::tf::get_datatype_attr(node, "dtype"));
  if (!(dtype == loco::DataType::S32 || dtype == loco::DataType::FLOAT32 ||
        dtype == loco::DataType::S8))
    return false;
  // TODO support other dtype

  return true;
}

void ConstGraphBuilder::build(const tensorflow::NodeDef &node, GraphBuilderContext *context) const
{
  loco::Graph *graph = context->graph();
  SymbolTable *tensor_names = context->tensor_names();

  // Create a "TFConstant" node for Const
  auto const_node = graph->nodes()->create<TFConst>();
  const_node->name(node.name());

  // set dtype
  auto dtype = plier::tf::as_loco_datatype(plier::tf::get_datatype_attr(node, "dtype"));
  const_node->dtype(dtype);

  // import shape and value
  const auto &input_tensor = plier::tf::get_tensor_attr(node, "value");
  const auto &input_shape = input_tensor.tensor_shape();
  const auto &input_dims = input_shape.dim();
  assert(input_shape.dim_size() <= 6);
  const_node->rank(input_shape.dim_size());
  int index = 0;
  bool zero_sized_shape = false;
  for (auto &d : input_dims)
  {
    assert(d.size() <= std::numeric_limits<int>::max());
    if (d.size() == 0)
      zero_sized_shape = true;

    assert(d.size() >= 0);
    const_node->dim(index++) = d.size();
  }

  int num_elements = 1;
  if (zero_sized_shape)
  {
    const_node->rank(0);
    num_elements = 0;
  }
  else
  {
    for (uint32_t d = 0; d < const_node->rank(); d++)
    {
      num_elements *= const_node->dim(d).value();
    }
  }

  switch (dtype)
  {
    case loco::DataType::S8:
      read_value_int8(const_node, num_elements, input_tensor);
      break;

    case loco::DataType::S32:
      read_value_int32(const_node, num_elements, input_tensor);
      break;

    case loco::DataType::FLOAT32:
      read_value_float32(const_node, num_elements, input_tensor);
      break;

      // TODO support other types

    default:
      assert(false);
  }

  // register string-name to node
  TensorName output_name(node.name(), 0);
  tensor_names->enroll(output_name, const_node);
}

} // namespace moco
