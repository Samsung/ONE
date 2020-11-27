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

#ifndef MIR_ONNX_ATTRIBUTE_HELPERS_H
#define MIR_ONNX_ATTRIBUTE_HELPERS_H

#include "onnx/onnx.pb.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mir_onnx
{

template <typename T> T getAttributeValue(const onnx::AttributeProto &attribute) = delete;

template <> inline float getAttributeValue(const onnx::AttributeProto &attribute)
{
  assert(attribute.type() == onnx::AttributeProto::FLOAT);
  return attribute.f();
}

template <> inline std::int64_t getAttributeValue(const onnx::AttributeProto &attribute)
{
  assert(attribute.type() == onnx::AttributeProto::INT);
  return attribute.i();
}

template <> inline std::string getAttributeValue(const onnx::AttributeProto &attribute)
{
  assert(attribute.type() == onnx::AttributeProto::STRING);
  return attribute.s();
}

template <> inline onnx::TensorProto getAttributeValue(const onnx::AttributeProto &attribute)
{
  assert(attribute.type() == onnx::AttributeProto::TENSOR);
  return attribute.t();
}

template <>
inline std::vector<std::int32_t> getAttributeValue(const onnx::AttributeProto &attribute)
{
  assert(attribute.type() == onnx::AttributeProto::INTS);
  // TODO Check that values fit.
  return {attribute.ints().cbegin(), attribute.ints().cend()};
}

template <>
inline std::vector<std::int64_t> getAttributeValue(const onnx::AttributeProto &attribute)
{
  assert(attribute.type() == onnx::AttributeProto::INTS);
  return {attribute.ints().cbegin(), attribute.ints().cend()};
}

inline const onnx::AttributeProto *findAttribute(const onnx::NodeProto &node,
                                                 const std::string &name)
{
  const auto &attributes = node.attribute();
  const auto it = std::find_if(
    attributes.cbegin(), attributes.cend(),
    [&name](const onnx::AttributeProto &attribute) { return attribute.name() == name; });
  if (it == attributes.cend())
    return nullptr;
  return &*it;
}

template <typename T> T getAttributeValue(const onnx::NodeProto &node, const std::string &name)
{
  const auto *attribute = findAttribute(node, name);
  if (attribute == nullptr)
    throw std::runtime_error("Cannot find attribute '" + name + "' in node '" + node.name() + "'.");
  return getAttributeValue<T>(*attribute);
}

template <typename T>
T getAttributeValue(const onnx::NodeProto &node, const std::string &name, T default_value)
{
  const auto *attribute = findAttribute(node, name);
  if (attribute == nullptr)
    return default_value;
  return getAttributeValue<T>(*attribute);
}

} // namespace mir_onnx

#endif // MIR_ONNX_ATTRIBUTE_HELPERS_H
