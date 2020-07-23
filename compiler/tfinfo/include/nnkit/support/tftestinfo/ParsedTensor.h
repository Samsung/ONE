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

#ifndef __NNKIT_SUPPORT_TFTESTINFO_PARSED_TENSOR_H__
#define __NNKIT_SUPPORT_TFTESTINFO_PARSED_TENSOR_H__

#include "nncc/core/ADT/tensor/Shape.h"

#include <oops/UserExn.h>

#include <string>
#include <cassert>
#include <stdexcept>

namespace nnkit
{
namespace support
{
namespace tftestinfo
{

/**
 * @brief Supported Data Types
 */
enum class DataType
{
  FLOAT32, // IEEE 32-bit floating point
  /* To be added */
};

/**
 * @brief Class to store tensor information parsed from test.info file under moco/test/tf
 */
class ParsedTensor final
{
public:
  enum class Kind
  {
    Input,
    Output
  };

  ParsedTensor() = delete;

  ParsedTensor(const Kind kind, const std::string &name, const DataType &dtype,
               const std::vector<int32_t> &shape)
      : _kind(kind), _dtype(dtype)
  {
    _tensor_name.assign(name);

    _shape.resize(shape.size());
    for (int rank = 0; rank < shape.size(); rank++)
      _shape.dim(rank) = shape.at(rank);
  }

  ~ParsedTensor()
  { /* empty */
  }

public:
  Kind kind() const { return _kind; }

  const std::string &name() { return _tensor_name; }

  const nncc::core::ADT::tensor::Shape &shape() const { return _shape; }
  // TODO This method is a bridge between testinfo and testinfo-v2. When testinfo-v2 is introduced,
  // this method will be removed.
  nncc::core::ADT::tensor::Shape &mutable_shape() { return _shape; }

  const DataType &dtype() const { return _dtype; }

  /**
   * @brief Get the name of node that has this tensor.
   * E.g., if the name of this tensor is "MyOp:0", this method returns "MyOp".
   */
  std::string nodeName() const { return _tensor_name.substr(0, _tensor_name.find(":")); }

  /**
   * @brief Get the index from the tensor name.
   * E.g., if the name of this tensor is "MyOp:12", this method returns 12.
   */
  int tensorIndex() const
  {
    int separator = _tensor_name.find(":");

    // sanity check
    if (separator == std::string::npos)
      throw oops::UserExn("Tensor name  in wrong format", "name", _tensor_name);

    return std::stoi(_tensor_name.substr(separator + 1, _tensor_name.length() - separator - 1));
  }

public:
  bool isFloatTensor() const { return _dtype == DataType::FLOAT32; }
  bool hasShape() const { return _has_shape; }

private:
  Kind _kind;
  std::string _tensor_name;
  nncc::core::ADT::tensor::Shape _shape;
  DataType _dtype;
  // Now, every info file has a shape.
  bool _has_shape = true;
};

} // namespace tftestinfo
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TFTESTINFO_PARSED_TENSOR_H__
