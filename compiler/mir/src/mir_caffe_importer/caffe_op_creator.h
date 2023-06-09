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

#ifndef MIR_CAFFE_OP_CREATOR_H
#define MIR_CAFFE_OP_CREATOR_H

#include <set>
#include <map>
#include <vector>
#include <memory>

#include "mir/Graph.h"
#include "mir/TensorVariant.h"
#include "mir/Shape.h"

#include "caffe/proto/caffe.pb.h"

namespace mir_caffe
{

class CaffeOpCreator
{
public:
  explicit CaffeOpCreator(mir::Graph *g) : _graph(g){};

  std::vector<mir::Operation::Output *> convertInput(const caffe::LayerParameter &layer);

  std::vector<mir::Operation::Output *>
  convertConvolution(const caffe::LayerParameter &layer,
                     const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertInnerProduct(const caffe::LayerParameter &layer,
                      const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertConcat(const caffe::LayerParameter &layer,
                const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertPooling(const caffe::LayerParameter &layer,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSoftmax(const caffe::LayerParameter &layer,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertReshape(const caffe::LayerParameter &layer,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertReLU(const caffe::LayerParameter &layer,
              const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertScale(const caffe::LayerParameter &layer,
               const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertBatchNorm(const caffe::LayerParameter &layer,
                   const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertDropout(const caffe::LayerParameter &layer,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertDeconvolution(const caffe::LayerParameter &layer,
                       const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertELU(const caffe::LayerParameter &layer,
             const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertEmbed(const caffe::LayerParameter &layer,
               const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSigmoid(const caffe::LayerParameter &layer,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertTanH(const caffe::LayerParameter &layer,
              const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertEltwise(const caffe::LayerParameter &layer,
                 const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertSplit(const caffe::LayerParameter &layer,
               const std::vector<mir::Operation::Output *> &inputs);

  std::vector<mir::Operation::Output *>
  convertLSTM(const caffe::LayerParameter &layer,
              const std::vector<mir::Operation::Output *> &inputs);

  void checkConvolution(const caffe::LayerParameter &layer,
                        std::set<std::string> &problems_ops_set);

  void checkPooling(const caffe::LayerParameter &layer, std::set<std::string> &problems_ops_set);

  void checkReshape(const caffe::LayerParameter &layer, std::set<std::string> &problems_ops_set);

  void checkBatchNorm(const caffe::LayerParameter &layer, std::set<std::string> &problems_ops_set);

  void checkLSTM(const caffe::LayerParameter &layer, std::set<std::string> &problems_ops_set);

private:
  mir::Graph *_graph = nullptr;

  std::vector<mir::Operation::Output *> createSplit(mir::Operation::Output *arg, int32_t num_parts,
                                                    int32_t axis);

  mir::Operation::Output *createFullyConnected(mir::Operation::Output *input,
                                               mir::Operation::Output *weights, int32_t axis);

  mir::TensorVariant convertBlob(const caffe::BlobProto &blob);

  template <typename OpType, typename... Types> mir::Operation *createOp(Types &&...args);
};

template <typename OpType, typename... Types>
mir::Operation *CaffeOpCreator::createOp(Types &&...args)
{
  return _graph->create<OpType>(std::forward<Types>(args)...);
}

} // namespace mir_caffe

#endif // MIR_CAFFE_OP_CREATOR_H
