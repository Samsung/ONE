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

#ifndef __MOCO_FRONTEND_ONNX_GRAPH_BUILDER_H__
#define __MOCO_FRONTEND_ONNX_GRAPH_BUILDER_H__

#include "GraphBuilderContext.h"

#include <onnx/onnx.pb.h>

namespace moco
{
namespace onnx
{

/**
 * @brief Parent class of onnx operation graph builders
 * @note GraphBuilder call proper build and validate function according to opset version
 */
class GraphBuilder
{
public:
  using OpsetVersion = int64_t;

  virtual bool validate(OpsetVersion, const ::onnx::NodeProto &) const { return true; }
  virtual void build(OpsetVersion, const ::onnx::NodeProto &, GraphBuilderContext *) const = 0;
  virtual ~GraphBuilder() {}
};

} // namespace onnx
} // namespace moco

#endif // __MOCO_FRONTEND_ONNX_GRAPH_BUILDER_H__
