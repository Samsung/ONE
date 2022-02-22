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

#include "Support.hpp"

#include <tensorflow/core/framework/graph.pb.h>

#include <memory>
#include <cassert>
#include <fstream>
#include <stdexcept>

namespace
{

template <typename T>
std::unique_ptr<T> open_fstream(const std::string &path, std::ios_base::openmode mode)
{
  if (path == "-")
  {
    return nullptr;
  }

  auto stream = std::make_unique<T>(path.c_str(), mode);
  if (!stream->is_open())
  {
    throw std::runtime_error{"Failed to open " + path};
  }
  return stream;
}

} // namespace

namespace tfkit
{
namespace tf
{

bool HasAttr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  return node.attr().count(attr_name) > 0;
}

tensorflow::DataType GetDataTypeAttr(const tensorflow::NodeDef &node, const std::string &attr_name)
{
  assert(HasAttr(node, attr_name));
  const auto &attr = node.attr().at(attr_name);
  assert(attr.value_case() == tensorflow::AttrValue::kType);
  return attr.type();
}

tensorflow::TensorProto *GetTensorAttr(tensorflow::NodeDef &node, const std::string &attr_name)
{
  assert(HasAttr(node, attr_name));
  tensorflow::AttrValue &attr = node.mutable_attr()->at(attr_name);
  assert(attr.value_case() == tensorflow::AttrValue::kTensor);
  return attr.mutable_tensor();
}

int GetElementCount(const tensorflow::TensorShapeProto &shape)
{
  int count = -1;

  for (auto &d : shape.dim())
  {
    if (d.size() == 0)
    {
      count = 0;
      break;
    }
    if (count == -1)
      count = 1;

    count *= d.size();
  }
  return count;
}

} // namespace tf

std::string CmdArguments::get(unsigned int index) const
{
  if (index >= _argc)
    throw std::runtime_error("Argument index out of bound");

  return std::string(_argv[index]);
}

std::string CmdArguments::get_or(unsigned int index, const std::string &s) const
{
  if (index >= _argc)
    return s;

  return std::string(_argv[index]);
}

std::unique_ptr<IOConfiguration> make_ioconfig(const CmdArguments &cmdargs)
{
  auto iocfg = std::make_unique<IOConfiguration>();

  auto in = open_fstream<std::ifstream>(cmdargs.get_or(0, "-"), std::ios::in | std::ios::binary);
  iocfg->in(std::move(in));

  auto out = open_fstream<std::ofstream>(cmdargs.get_or(1, "-"), std::ios::out | std::ios::binary);
  iocfg->out(std::move(out));

  return iocfg;
}

} // namespace tfkit
