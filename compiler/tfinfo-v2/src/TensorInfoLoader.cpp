/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tfinfo-v2/TensorInfoLoader.h"

#include "tfinfo-v2/TensorSignature.h"

#include <oops/UserExn.h>
#include <stdex/Memory.h>

#include <tfinfo-v2.pb.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <fcntl.h>

namespace
{

// for testing purpose
bool load_text(std::istream *stream, tfinfo_v2_proto::InfoDef &info_def)
{
  google::protobuf::io::IstreamInputStream iis(stream);

  return google::protobuf::TextFormat::Parse(&iis, &info_def);
}

bool is_num(const std::string &num)
{
  for (int i = 0; i < num.length(); i++)
    if (!isdigit(num[i]))
      return false;

  return true;
}

void validate_tensor_name(const std::string &tensor_name, const char *path)
{
  // Note that Tensorflow tensor name format is
  // operation name ":" index, e.g., "in/placeholder:0"
  int pos = tensor_name.find(":");
  if (pos == std::string::npos)
    throw oops::UserExn("Missing index separator, ':'", "name", tensor_name, "file", path);

  if (tensor_name.length() == pos + 1) // ':' is  the last char
    throw oops::UserExn("Missing tensor index after ':'", "name", tensor_name, "file", path);

  // 1. Validating operation name.
  // for naming format, refer to https://www.tensorflow.org/api_docs/python/tf/Operation#__init__
  // First char is in the form of "[A-Za-z0-9.]"
  // and the rest chars are  in the form of "[A-Za-z0-9_.\\-/]*"
  std::string op_name = tensor_name.substr(0, pos);

  // first character
  if (!(isalnum(op_name[0]) || op_name[0] == '.'))
    throw oops::UserExn("Wrong tensor name format", "name", tensor_name, "file", path);

  // and the rest chars
  for (int i = 1; i < op_name.length(); i++)
    if (!(isalnum(op_name[i]) || std::string("_.\\-/").find(op_name[i]) != std::string::npos))
      throw oops::UserExn("Wrong tensor name format", "name", tensor_name, "file", path);

  // 2. validating index after ":"
  std::string index = tensor_name.substr(pos + 1, op_name.length() - pos - 1);

  if (!is_num(index))
    throw oops::UserExn("Wrong tensor name format", "name", tensor_name, "file", path);
}

void check_duplicate(tfinfo::v2::TensorSignatures &tensors, const char *path)
{
  std::map<std::string, bool> tool;
  for (auto &tensor : tensors)
  {
    if (tool.find(tensor->name()) != tool.end())
      throw oops::UserExn("Duplicate tensor definition", "name", tensor->name(), "file", path);
    else
      tool[tensor->name()] = true;
  }
}

void convert(tfinfo_v2_proto::InfoDef &info_def, tfinfo::v2::TensorSignatures &tensors,
             const char *path)
{
  // processing input. Note that there could be no input.
  if (auto input_size = info_def.input_size())
  {
    for (int i = 0; i < input_size; i++)
    {
      auto input_def = info_def.input().Get(i);

      auto name = input_def.name();
      validate_tensor_name(name, path);

      auto tensor = stdex::make_unique<tfinfo::v2::TensorSignature>(
        tfinfo::v2::TensorSignature::Kind::Input, name);

      // when there is dim attribute for unknown shape
      if (input_def.dim_size() > 0)
      {
        for (int d = 0; d < input_def.dim().size(); d++)
        {
          auto dim = input_def.dim(d);
          tensor->shapeHint().add(dim.axis(), dim.size());
        }
      }

      tensors.emplace_back(std::move(tensor));
    }
  }

  // processing output
  auto output_size = info_def.output_size();
  if (output_size == 0)
    throw oops::UserExn("Missing output node. At least 1 output node must exist", "file", path);

  if (auto output_node_size = info_def.output_size())
  {
    for (int i = 0; i < output_node_size; i++)
    {
      auto name = info_def.output().Get(i).name();
      validate_tensor_name(name, path);

      auto tensor = stdex::make_unique<tfinfo::v2::TensorSignature>(
        tfinfo::v2::TensorSignature::Kind::Output, name);
      tensors.emplace_back(std::move(tensor));
    }
  }

  check_duplicate(tensors, path);
}

} // namespace

namespace tfinfo
{
inline namespace v2
{

TensorSignatures load(const char *path)
{
  std::ifstream stream(path, std::ios::in | std::ios::binary);

  return load(&stream, path);
}

TensorSignatures load(std::istream *stream, const char *path_for_error_msg)
{
  tfinfo_v2_proto::InfoDef info_def;

  if (!load_text(stream, info_def))
  {
    throw oops::UserExn("Cannot parse the info file", "path", path_for_error_msg);
  }

  TensorSignatures tensors;

  convert(info_def, tensors, path_for_error_msg);

  return tensors;
}

} // namespace v2
} // namespace tfinfo
