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

#include "GraphBuilder.h"

#include <cassert>

namespace moco
{
namespace onnx
{

/**
 * @brief GraphBuilder for Constant(since version 1) node
 */
class Constant_V1
{
public:
  bool validate(const ::onnx::NodeProto &) const;
  void build(const ::onnx::NodeProto &, GraphBuilderContext *) const;
};

/**
 * @brief GraphBuilder for Constant(since version 9) node
 * @note Until version 1, only FLOAT16, FLOAT, DOUBLE was supported
 *       Since version 9, all types are supported
 */
class Constant_V9
{
public:
  bool validate(const ::onnx::NodeProto &) const;
  void build(const ::onnx::NodeProto &, GraphBuilderContext *) const;
};

/**
 * @brief GraphBuilder for Constant node
 */
class ConstantGraphBuilder : public GraphBuilder
{
public:
  bool validate(OpsetVersion, const ::onnx::NodeProto &) const;
  void build(OpsetVersion, const ::onnx::NodeProto &, GraphBuilderContext *) const;
};

} // namespace onnx
} // namespace moco
