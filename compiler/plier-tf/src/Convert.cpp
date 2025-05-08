/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <plier/tf/Convert.h>

#include <nncc/core/ADT/tensor/Shape.h>

#include <cassert>
#include <stdexcept>

namespace plier
{
namespace tf
{

bool has_attr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  return node.attr().count(attr_name) > 0;
}

bool has_attrs(const tensorflow::NodeDef &node, const std::vector<std::string> &attr_names)
{
  for (auto &attr : attr_names)
    if (!has_attr(node, attr))
      return false;
  return true;
}

tensorflow::DataType get_datatype_attr(const tensorflow::NodeDef &node,
                                       const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kType);
  return node.attr().at(attr_name).type();
}

const tensorflow::TensorShapeProto &get_shape_attr(const tensorflow::NodeDef &node,
                                                   const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kShape);
  return node.attr().at(attr_name).shape();
}

const tensorflow::TensorProto &get_tensor_attr(const tensorflow::NodeDef &node,
                                               const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kTensor);
  return node.attr().at(attr_name).tensor();
}

const ::tensorflow::AttrValue_ListValue &get_list_attr(const tensorflow::NodeDef &node,
                                                       const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kList);
  return node.attr().at(attr_name).list();
}

const std::string &get_string_attr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kS);
  return node.attr().at(attr_name).s();
}

int64_t get_int_attr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kI);
  return node.attr().at(attr_name).i();
}

float get_float_attr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kF);
  return node.attr().at(attr_name).f();
}

bool get_bool_attr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  assert(has_attr(node, attr_name));
  assert(node.attr().at(attr_name).value_case() == tensorflow::AttrValue::kB);
  return node.attr().at(attr_name).b();
}

std::vector<int64_t> as_int64_list(const tensorflow::AttrValue_ListValue &lv)
{
  std::vector<int64_t> vi;
  int isize = lv.i_size();

  vi.resize(isize);
  for (int i = 0; i < isize; ++i)
    vi[i] = lv.i(i);

  return vi;
}

loco::DataType as_loco_datatype(const tensorflow::DataType tf_dtype)
{
  switch (tf_dtype)
  {
    case tensorflow::DT_INT8:
      return loco::DataType::S8;
    case tensorflow::DT_UINT8:
      return loco::DataType::U8;
    case tensorflow::DT_FLOAT:
      return loco::DataType::FLOAT32;
    case tensorflow::DT_INT32:
      return loco::DataType::S32;
    case tensorflow::DT_INT64:
      return loco::DataType::S64;
    case tensorflow::DT_BOOL:
    case tensorflow::DT_STRING:
    case tensorflow::DT_COMPLEX64:
    default:
      break;
  }
  throw std::runtime_error{"Unsupported tensorflow dtype: " + tensorflow::DataType_Name(tf_dtype)};
}

DataLayout as_data_layout(const std::string &tf_layout_str)
{
  if (tf_layout_str == "NHWC")
    return DataLayout::NHWC;
  else if (tf_layout_str == "NCHW")
    return DataLayout::NCHW;
  else
    throw std::runtime_error("unknown data layout");
}

DataLayout get_data_layout(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  auto layout = get_string_attr(node, attr_name);

  if (layout == "NHWC")
    return DataLayout::NHWC;
  else if (layout == "NCHW")
    return DataLayout::NCHW;
  else
    throw std::runtime_error("unknown data layout");
}

void copy_shape(const tensorflow::TensorShapeProto &tf_shape,
                nncc::core::ADT::tensor::Shape &to_shape)
{
  assert(!tf_shape.unknown_rank());

  int64_t tf_rank = tf_shape.dim_size();
  assert(tf_rank < std::numeric_limits<uint32_t>::max());

  int32_t rank = static_cast<int32_t>(tf_rank);
  to_shape.resize(rank);

  for (int32_t d = 0; d < rank; d++)
  {
    int64_t dim_value = tf_shape.dim(d).size();
    assert(dim_value < std::numeric_limits<uint32_t>::max());

    if (dim_value >= 0LL)
    {
      uint32_t dim_value32 = static_cast<uint32_t>(dim_value);
      to_shape.dim(d) = dim_value32;
    }
    else
    {
      throw std::runtime_error("Cannot handle unknown dimension");
      // TODO support unknown dimension
    }
  }
}

} // namespace tf
} // namespace plier
