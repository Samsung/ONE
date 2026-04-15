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

#ifndef MIR_TFLITE_OP_CREATOR_H
#define MIR_TFLITE_OP_CREATOR_H

#include "schema_generated.h"

#include "mir/Graph.h"

#include <utility>
#include <vector>

namespace mir_tflite
{

class TFLiteOpCreator
{
public:
  explicit TFLiteOpCreator(mir::Graph *g) : _graph(g) {}

  std::vector<mir::Operation::Output *>
  convertConv2D(const tflite::Conv2DOptionsT *opts,
                const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertDepthwiseConv2D(const tflite::DepthwiseConv2DOptionsT *opts,
                         const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertConcatenation(const tflite::ConcatenationOptionsT *opts,
                       const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertMaxPool2D(const tflite::Pool2DOptionsT *opts,
                   const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertAveragePool2D(const tflite::Pool2DOptionsT *opts,
                       const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertMean(const tflite::ReducerOptionsT *opts,
              const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSoftmax(const tflite::SoftmaxOptionsT *opts,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSlice(const tflite::SliceOptionsT *opts,
               const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertReshape(const tflite::ReshapeOptionsT *opts,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertFullyConnected(const tflite::FullyConnectedOptionsT *opts,
                        const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertResizeNearestNeighbor(const tflite::ResizeNearestNeighborOptionsT *opts,
                               const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertLogistic(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertRsqrt(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSqrt(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSqueeze(const tflite::SqueezeOptionsT *opts,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertAdd(const tflite::AddOptionsT *opts, const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSub(const tflite::SubOptionsT *opts, const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertMul(const tflite::MulOptionsT *opts, const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertDiv(const tflite::DivOptionsT *opts, const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertMax(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSquaredDifference(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertTanh(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertReLU(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertReLU6(const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertTransposeConv(const tflite::TransposeConvOptionsT *opts,
                       const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertPad(const tflite::PadOptionsT *opts, const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertTranspose(const tflite::TransposeOptionsT *opts,
                   const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertStridedSlice(const tflite::StridedSliceOptionsT *opts,
                      const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertLeakyReLU(const tflite::LeakyReluOptionsT *opts,
                   const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertShape(const tflite::ShapeOptionsT *opts,
               const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertHardSwish(const tflite::HardSwishOptionsT *opts,
                   const std::vector<mir::Operation::Output *> &inputs);

private:
  mir::Graph *_graph;

  mir::Operation::Output *addFusedActivation(mir::Operation::Output *input,
                                             tflite::ActivationFunctionType activation_type);

  template <typename OpType, typename... Types> mir::Operation *createOp(Types &&...args);
};

template <typename OpType, typename... Types>
mir::Operation *TFLiteOpCreator::createOp(Types &&...args)
{
  return _graph->create<OpType>(std::forward<Types>(args)...);
}

} // namespace mir_tflite

#endif // MIR_TFLITE_OP_CREATOR_H
