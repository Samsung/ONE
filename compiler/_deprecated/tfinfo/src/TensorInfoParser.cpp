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

#include "nnkit/support/tftestinfo/TensorInfoParser.h"
#include "nnkit/support/tftestinfo/ParsedTensor.h"

// TODO Remove this file after code cleanup
#include "Compat.h"

#include <oops/UserExn.h>
#include <nncc/core/ADT/tensor/Shape.h>

#include <cctype>
#include <memory>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>

/**
 * @file     TensorInfoParser.cpp
 * @brief    This file contains functions to parse test.info files in moco/test/tf
 */

namespace
{

using nnkit::support::tftestinfo::ParsedTensor;

// remove comment
void remove_comment(std::string &line)
{
  int pos = line.find_first_of("#");
  if (pos != std::string::npos)
    line.erase(pos);
}

std::string trim(const std::string &str)
{
  static const std::string whitespace = " \t";
  static const std::string empty = "";

  const auto begin = str.find_first_not_of(whitespace);
  if (begin == std::string::npos)
    return empty;

  const auto end = str.find_last_not_of(whitespace);

  return str.substr(begin, end - begin + 1);
}

ParsedTensor::Kind get_kind(const std::string &tok)
{
  if (tok == "input")
    return ParsedTensor::Kind::Input;
  else if (tok == "output")
    return ParsedTensor::Kind::Output;
  else
    throw oops::UserExn("Unrecognizable token", "token", tok);
}

TF_DataType get_dtype(const std::string &tok)
{
  if (tok == "TF_FLOAT")
    return TF_FLOAT;
  else
    throw oops::UserExn("Unsupported tensor datatype", "data type", tok);
}

bool validate_num(const std::string &num)
{
  for (int i = 0; i < num.length(); i++)
    if (!isdigit(num[i]))
      return false;

  return true;
}

bool validate_name(const std::string &tensor_name)
{
  // Note that Tensorflow tensor name format is
  // operation name ":" index, e.g., "in/placeholder:0"
  int pos = tensor_name.find(":");
  if (pos == std::string::npos)
    return false;

  if (tensor_name.length() == pos + 1) // ':' is  the last char
    return false;

  // 1. Validating operation name.
  // for naming format, refer to https://www.tensorflow.org/api_docs/python/tf/Operation#__init__
  // First char is in the form of "[A-Za-z0-9.]"
  // and the rest chars are  in the form of "[A-Za-z0-9_.\\-/]*"
  std::string op_name = tensor_name.substr(0, pos);

  // first character
  if (!(isalnum(op_name[0]) || op_name[0] == '.'))
    return false;

  // and the rest chars
  for (int i = 1; i < op_name.length(); i++)
    if (!(isalnum(op_name[i]) || std::string("_.\\-/").find(op_name[i]) != std::string::npos))
      return false;

  // 2. validating index after ":"
  std::string index = tensor_name.substr(pos + 1, op_name.length() - pos - 1);

  return validate_num(index);
}

} // namespace

namespace nnkit
{
namespace support
{
namespace tftestinfo
{

#define CHECK_NOT_NULL(x) \
  if (!(x))               \
  oops::UserExn("Cannot find required token")

/**
 * @brief Function to parse a line of test.info file
 * Examples:
 *   - "input, in/placeholder_32:0, TF_INT32, [3, 4, 2, 3]"
 *   - "output, result:0, TF_FLOAT, []"
 */
std::unique_ptr<ParsedTensor> parse_line(std::string &line)
{
  // parsed data
  ParsedTensor::Kind kind;
  std::string name;
  TF_DataType dtype;
  std::vector<int32_t> shape;

  remove_comment(line);

  if (line.length() == 0) // empty line or line with comment
    return nullptr;

  std::string tok, trimmed, dim;

  std::istringstream line_stream(line);

  CHECK_NOT_NULL(std::getline(line_stream, tok, ',')); // kind
  kind = get_kind(trim(tok));

  CHECK_NOT_NULL(std::getline(line_stream, tok, ',')); // tensor name
  trimmed = trim(tok);
  if (!validate_name(trimmed))
    throw oops::UserExn("Tensor name in wrong format", "name", tok);
  name.assign(trimmed);

  CHECK_NOT_NULL(std::getline(line_stream, tok, ',')); // data type
  dtype = get_dtype(trim(tok));

  CHECK_NOT_NULL(std::getline(line_stream, tok, '[')); // start of shape
  trimmed = trim(tok);
  if (trimmed.length())
    throw oops::UserExn("Unknown token between data type and shape", "token", tok);

  CHECK_NOT_NULL(std::getline(line_stream, tok, ']'));

  std::istringstream shape_stream(tok);

  bool first = true;
  while (std::getline(shape_stream, dim, ',')) // each dim
  {
    dim = trim(dim);

    if (first && dim.length() == 0)
      continue; // scalar
    first = false;

    if (dim.length() == 0)
      throw oops::UserExn("Empty dim in shape", "shape", tok);

    if (!validate_num(dim))
      throw oops::UserExn("Dim in shape must be a number", "dim", dim);

    shape.emplace_back(std::stoi(dim));
  }

  return std::make_unique<ParsedTensor>(kind, name, dtype, shape);
}

#undef CHECK_NOT_NULL

std::vector<std::unique_ptr<ParsedTensor>> parse(const char *info_path)
{
  std::ifstream infile;
  infile.open(info_path);

  if (infile.fail())
  {
    throw oops::UserExn("Fail to open file", "path", info_path);
  }

  std::vector<std::unique_ptr<ParsedTensor>> tensors;

  std::string line;
  while (std::getline(infile, line))
  {
    auto tensor = parse_line(line);
    if (tensor)
      tensors.emplace_back(std::move(tensor));
  }

  return tensors;
}

} // namespace tftestinfo
} // namespace support
} // namespace nnkit
