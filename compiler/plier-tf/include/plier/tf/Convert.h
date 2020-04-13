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

#ifndef __PLIER_TF_CONVERT_H__
#define __PLIER_TF_CONVERT_H__

#include <loco.h>
#include <nncc/core/ADT/tensor/Shape.h>

#include <tensorflow/core/framework/graph.pb.h>

#include <string>

namespace plier
{
namespace tf
{

bool has_attr(const tensorflow::NodeDef &node, const std::string &attr_name);
bool has_attrs(const tensorflow::NodeDef &node, const std::vector<std::string> &attr_names);

tensorflow::DataType get_datatype_attr(const tensorflow::NodeDef &node,
                                       const std::string &attr_name);
const tensorflow::TensorShapeProto &get_shape_attr(const tensorflow::NodeDef &node,
                                                   const std::string &attr_name);
const tensorflow::TensorProto &get_tensor_attr(const tensorflow::NodeDef &node,
                                               const std::string &attr_name);
const tensorflow::AttrValue_ListValue &get_list_attr(const tensorflow::NodeDef &node,
                                                     const std::string &attr_name);
const std::string &get_string_attr(const tensorflow::NodeDef &node, const std::string &attr_name);
int64_t get_int_attr(const tensorflow::NodeDef &node, const std::string &attr_name);
float get_float_attr(const tensorflow::NodeDef &node, const std::string &attr_name);
bool get_bool_attr(const tensorflow::NodeDef &node, const std::string &attr_name);

std::vector<int64_t> as_int64_list(const tensorflow::AttrValue_ListValue &lv);

loco::DataType as_loco_datatype(const tensorflow::DataType dtype);

/**
 * @brief Class to represent TensorFlow "data_format" attr.
 */
enum class DataLayout
{
  NHWC,
  NCHW,
};

/// @ brief Convert TF Data Layout string (e.g., "NHWC") to enum class for programming convenience
DataLayout as_data_layout(const std::string &tf_layout_str);

DataLayout get_data_layout(const tensorflow::NodeDef &node, const std::string &attr_name);

/**
 * @brief Copy shape defined in TensorShapeProto to angkor shape
 *
 * @note Unknown dimension is not supported
 */
void copy_shape(const tensorflow::TensorShapeProto &tf_shape,
                nncc::core::ADT::tensor::Shape &to_shape);

} // namespace tf
} // namespace plier

#endif // __PLIER_TF_CONVERT_H__
