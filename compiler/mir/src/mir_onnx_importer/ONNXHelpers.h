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

#ifndef __MIR_ONNX_HELPERS_H__
#define __MIR_ONNX_HELPERS_H__

#include "mir/Graph.h"
#include "mir/ops/ConstantOp.h"
#include "mir/TensorVariant.h"
#include "mir/ops/TransposeOp.h"

#include "onnx/onnx.pb.h"

namespace mir_onnx
{

extern const int64_t firstUnknownOpset;

mir::DataType onnxDataTypeToMirDataType(onnx::TensorProto::DataType type);

mir::Shape constantToShape(const mir::ops::ConstantOp *op);

mir::TensorVariant createTensor(const onnx::TensorProto *tensor);

mir::Operation *foldConstants(mir::Graph *graph, mir::Operation *op);

template <typename OpType, typename... Types>
mir::Operation *createOp(mir::Graph *graph, Types &&...args)
{
  auto op = graph->create<OpType>(std::forward<Types>(args)...);
  op = foldConstants(graph, op);
  return op;
}

} // namespace mir_onnx

#endif // __MIR_ONNX_HELPERS_H__
