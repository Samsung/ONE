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

#ifndef _NNC_SOFT_BACKEND_MODEL_ANALYZER_H_
#define _NNC_SOFT_BACKEND_MODEL_ANALYZER_H_

#include "SequencedIR.h"

#include "mir/Graph.h"
#include "mir/Visitor.h"
#include "mir/Shape.h"
#include "mir/TensorVariant.h"
#include "mir/Operation.h"

#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <iterator>

namespace nnc
{

/**
 * @brief Constructs inference sequence for given computational graph,
 * gathers list of variables used in artifact.
 */
class ModelAnalyzer : public mir::Visitor
{
public:
  /**
   * @brief contructs inference sequence
   * @param g pointer to graph to linearize
   */
  void analyze(const mir::Graph *g);

  void visit(mir::ops::AbsOp &) override;
  void visit(mir::ops::AddOp &op) override;
  void visit(mir::ops::AvgPool2DOp &op) override;
  void visit(mir::ops::BroadcastOp &op) override;
  void visit(mir::ops::CappedReluOp &op) override;
  void visit(mir::ops::ConcatOp &op) override;
  void visit(mir::ops::ConstantOp &op) override;
  void visit(mir::ops::Conv2DOp &op) override;
  void visit(mir::ops::DeConv2DOp &op) override;
  void visit(mir::ops::DepthwiseConv2DOp &op) override;
  void visit(mir::ops::DivOp &op) override;
  void visit(mir::ops::EluOp &op) override;
  void visit(mir::ops::FullyConnectedOp &op) override;
  void visit(mir::ops::GatherOp &op) override;
  void visit(mir::ops::InputOp &op) override;
  void visit(mir::ops::LeakyReluOp &op) override;
  void visit(mir::ops::MaxOp &op) override;
  void visit(mir::ops::MaxPool2DOp &op) override;
  void visit(mir::ops::MulOp &op) override;
  void visit(mir::ops::OutputOp &op) override;
  void visit(mir::ops::PadOp &op) override;
  void visit(mir::ops::ReduceMeanOp &op) override;
  void visit(mir::ops::ReluOp &op) override;
  void visit(mir::ops::ReshapeOp &op) override;
  void visit(mir::ops::ResizeOp &op) override;
  void visit(mir::ops::SigmoidOp &op) override;
  void visit(mir::ops::SliceOp &op) override;
  void visit(mir::ops::SoftmaxOp &op) override;
  void visit(mir::ops::SqrtOp &op) override;
  void visit(mir::ops::SqueezeOp &op) override;
  void visit(mir::ops::SubOp &op) override;
  void visit(mir::ops::TanhOp &op) override;
  void visit(mir::ops::TransposeOp &op) override;

  /**
   * @return vector of id's of network input tensors
   */
  const std::vector<size_t> &getInputs() const { return _inputs; }

  /**
   * @return vector of id's of tensors with unique names taken from Model IR
   */
  const std::vector<size_t> &getPersistentTensors() const { return _persistent_tensors; }

  /**
   * @return vector of id's of network output tensors
   */
  const std::vector<size_t> &getOutputs() const { return _outputs; }

  /**
   * @return vector of all network tensors
   */
  const std::vector<sir::TensorDescriptor> &getTensors() const { return _tensors; }

  /**
   * @return Inference sequence
   */
  const std::vector<std::unique_ptr<sir::Action>> &getInferenceSequence() const
  {
    return _inferenceSequence;
  }

  /**
   * @return Inference sequence
   */
  std::vector<std::unique_ptr<sir::Action>> &getInferenceSequence() { return _inferenceSequence; }

  /**
   * @return Model name, taken from Model IR
   */
  const std::string &getModelName() const { return _modelName; }

  size_t getMaxTemporarySize() const { return _max_temp_size; }

  size_t getTempTID() const { return _temp_tensor_id; }

protected:
  void visit_fallback(mir::Operation &op) override;

private:
  /**
   * @brief Common function to add function call in inference sequence
   * @param op Node representing added call
   * @param function_name Function name
   * @param aux_args Auxilliary argument ids
   *
   * Inserts information about CG operation into inference sequence: name of operation,
   * creates tensors for operation outputs, binds operation inputs with tensors from previous
   * operations
   */
  void appendOperationToInference(mir::Operation *op, const std::string &function_name,
                                  std::vector<size_t> aux_args = {});

  /**
   * @brief Registers a temporary buffer of size *size* used by op *op_id*
   * @param size Size of buffer
   */
  void updateMaxTemporarySize(size_t size);

  /**
   * @brief Declares input tensor in artifact
   * @param name Name of tensor
   * @param shape expected shape of input
   * @return Id of created tensor
   */
  size_t declareInputTensor(const std::string &name, const mir::Shape &shape);

  /**
   * @brief Declares persistent tensor in artifact
   * @param name Name of variable, if empty - assigned automaticly
   * @return Id of created tensor
   */
  size_t declarePersistentTensor(const std::string &name);

  /**
   * @brief Declares temporary tensor in artifact
   * @return Id of created tensor
   */
  size_t declareTemporaryTensor();

  /**
   * @brief Gathers info where tensors were defined and used in inference sequence
   * @param sequence Sequence of operations in inference
   * @param first_def Maps tensor id to position in inf sequence where it was defined first time.
   * @param last_use Maps tensor id to position in inf sequence where it was used last time.
   */
  void gatherDefUseInfo(const std::vector<std::unique_ptr<sir::Action>> &post_order,
                        std::map<size_t, size_t> &first_def, std::map<size_t, size_t> &last_use);

  /**
   * @brief constructs inference sequence from vector of mir::Operations, constructed
   * @param post_order vector representing layout of operations in inference
   */
  void constructInferenceSequence(const std::vector<mir::Operation *> &post_order);

  /**
   * @brief Fill list of outputs in ModelAnalyzer
   * @param g Graph where to get list of outputs
   */
  void collectOutputs(const mir::Graph *g);

  std::string _modelName = "NN";
  std::vector<std::unique_ptr<sir::Action>> _inferenceSequence;
  size_t _allocatedTensors = 0;
  std::vector<size_t> _inputs;
  /// @brief list of persistent tensors
  std::vector<size_t> _persistent_tensors;
  /// @brief list of tensor ids corresponding to NN outputs
  std::vector<size_t> _outputs;
  size_t _max_temp_size = 0;
  size_t _temp_tensor_id = 0;
  std::vector<sir::TensorDescriptor> _tensors;
  std::map<const mir::Operation *, const sir::Action *> _opToDescr;
};

} // namespace nnc

#endif //_NNC_SOFT_BACKEND_MODEL_ANALYZER_H_
