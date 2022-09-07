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

#include "CustomopConfLoader.h"

#include <loco.h>
#include <cwrap/Fildes.h>
#include <angkor/TensorShape.h>

#include <CustomOpInfo.pb.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <limits> // std::numeric_limits

#include <fcntl.h>

namespace
{
bool load_text(const cwrap::Fildes &fildes, tf2tflite::CustomOpInfoDef &def)
{
  google::protobuf::io::FileInputStream fis(fildes.get());

  return google::protobuf::TextFormat::Parse(&fis, &def);
}

angkor::TensorShape convert_shape(const tf2tflite::ShapeProto &shape)
{
  angkor::TensorShape to_shape;

  int64_t rank64 = shape.dim_size();
  assert(rank64 < std::numeric_limits<uint32_t>::max());

  int32_t rank = static_cast<int32_t>(rank64);
  to_shape.resize(rank);

  for (int32_t d = 0; d < rank; d++)
  {
    int64_t dim_value = shape.dim(d).size();
    assert(dim_value >= 0ULL);
    assert(dim_value < std::numeric_limits<uint32_t>::max());

    uint32_t dim_value32 = static_cast<uint32_t>(dim_value);
    to_shape.dim(d) = dim_value32;
  }

  return to_shape;
}

loco::DataType convert_dtype(const tf2tflite::DataType &dtype)
{
  if (dtype == tf2tflite::DT_FLOAT)
    return loco::DataType::FLOAT32;
  else if (dtype == tf2tflite::DT_INT32)
    return loco::DataType::S32;
  else
    throw std::runtime_error("Not yet supported datatype. Cannot convert.");
}

// Note : the following functions look similar with plier::tf::Convert.h.
// However, the schema is different.(not "tensorflow::..." but "tf2tflite::...")
// So, plier::tf cannot be used.
loco::DataType get_dtype_attr(const tf2tflite::CustomOpDef &custom_op)
{
  std::string type_attr_name("dtype");

  assert(custom_op.attr().count(type_attr_name) > 0);
  const auto &attr = custom_op.attr().at(type_attr_name);
  assert(attr.value_case() == tf2tflite::AttrValue::kType);
  auto dtype_def = attr.type();

  return convert_dtype(dtype_def);
}

angkor::TensorShape get_shape_attr(const tf2tflite::CustomOpDef &custom_op)
{
  std::string shape_attr_name("output_shape");

  assert(custom_op.attr().count(shape_attr_name) > 0);
  const auto &attr = custom_op.attr().at(shape_attr_name);
  assert(attr.value_case() == tf2tflite::AttrValue::kShape);
  auto shape_def = attr.shape();

  return convert_shape(shape_def);
}

void add_customop(tf2tflite::CustomOpInfoDef &def, moco::ModelSignature &sig)
{
  for (const auto &custom_op : def.custom_op())
  {
    sig.add_customop(custom_op.op());

    auto name = custom_op.name();

    // setting dtype and shape to ModelSignature
    sig.dtype(name, get_dtype_attr(custom_op));
    sig.shape(name, get_shape_attr(custom_op));
  }
}

} // namespace

namespace tf2tflite
{

void load_customop_conf(const std::string &path, moco::ModelSignature &sig)
{
  CustomOpInfoDef def;

  cwrap::Fildes fildes{open(path.c_str(), O_RDONLY)};

  if (fildes.get() < 0)
  {
    throw std::runtime_error{"Error: " + path + " not found"};
  }

  if (!load_text(fildes, def))
  {
    throw std::runtime_error{"Error: Failed to parse prototxt " + path};
  }

  add_customop(def, sig);
}

} // namespace tf2tflite
