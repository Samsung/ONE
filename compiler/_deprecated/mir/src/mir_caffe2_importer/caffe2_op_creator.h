/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef MIR_CAFFE2_OP_CREATOR_H
#define MIR_CAFFE2_OP_CREATOR_H

#include <set>
#include <unordered_map>
#include <vector>
#include <memory>

#include "mir/Graph.h"
#include "mir/Operation.h"
#include "mir/TensorVariant.h"
#include "mir/Shape.h"

#include "caffe2/proto/caffe2.pb.h"

namespace mir_caffe2
{

using mir::Operation;
using mir::Shape;

class Caffe2OpCreator
{
public:
  explicit Caffe2OpCreator(mir::Graph *g) : _graph(g) {}

  std::vector<mir::Operation::Output *>
  convertConstant(const std::vector<mir::Operation::Output *> &inputs,
                  const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertAdd(const std::vector<mir::Operation::Output *> &inputs, const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertAveragePool(const std::vector<mir::Operation::Output *> &inputs,
                     const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertConv(const std::vector<mir::Operation::Output *> &inputs, const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertConcat(const std::vector<mir::Operation::Output *> &inputs,
                const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertDropout(const std::vector<mir::Operation::Output *> &inputs,
                 const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertFC(const std::vector<mir::Operation::Output *> &inputs, const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertMaxPool(const std::vector<mir::Operation::Output *> &inputs,
                 const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertMul(const std::vector<mir::Operation::Output *> &inputs, const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertRelu(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertResizeNearest(const std::vector<mir::Operation::Output *> &inputs,
                       const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertSigmoid(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSoftmax(const std::vector<mir::Operation::Output *> &inputs,
                 const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertSpatialBN(const std::vector<mir::Operation::Output *> &inputs,
                   const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertSum(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertClip(const std::vector<mir::Operation::Output *> &inputs, const ::caffe2::OperatorDef &op);

  std::vector<mir::Operation::Output *>
  convertReshape(const std::vector<mir::Operation::Output *> &inputs,
                 const ::caffe2::OperatorDef &op);

private:
  mir::Graph *_graph = nullptr;

  template <typename OpType, typename... Types> mir::Operation *createOp(Types &&...args);
};

template <typename OpType, typename... Types>
mir::Operation *Caffe2OpCreator::createOp(Types &&...args)
{
  return _graph->create<OpType>(std::forward<Types>(args)...);
}

} // namespace mir_caffe2

#endif // MIR_CAFFE2_OP_CREATOR_H
