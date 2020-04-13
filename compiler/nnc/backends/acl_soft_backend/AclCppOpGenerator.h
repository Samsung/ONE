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

#ifndef _NNC_ACL_CPP_OP_GENERATOR_H_
#define _NNC_ACL_CPP_OP_GENERATOR_H_

#include "mir/Visitor.h"
#include "mir/TensorVariant.h"
#include "mir/Operation.h"
#include "mir/Graph.h"
#include "ArtifactModel.h"
#include "ArtifactGeneratorCppCode.h"
#include "ArtifactGeneratorCppDecl.h"

#include <set>

namespace nnc
{

/**
 * @brief Implements the visitor for the model IR which generates the DOM description
 *        translated to C++ source/header files by the ACL soft backend code generators.
 */
class AclCppOpGenerator : public mir::Visitor
{
public:
  AclCppOpGenerator(const std::string &name, std::ostream &par_out);
  /**
   * @brief The main interface function to the class. Convers the model IR to the DOM.
   * @param g - pointer the model IR graph.
   * @return - reference to the top-level DOM entity.
   */
  const ArtifactModule &generate(mir::Graph *g);

  /**
   * @brief Implementations of the MIR visitors.
   * @param op
   */
  void visit(mir::ops::AddOp &op) override;
  void visit(mir::ops::AvgPool2DOp &op) override;
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

protected:
  void visit_fallback(mir::Operation &op) override;

private:
  using AF = ArtifactFactory;

  /**
   * @brief generate transpose of input tensor NHWC -> NCHW
   * @param name name of tensor containing transposed data
   * @param input_shape shape of @p input
   * @param input id of input tensor
   * @return Id of result tensor
   */
  std::shared_ptr<ArtifactId> genTransposeMIRtoACL(const std::string &name,
                                                   const mir::Shape &input_shape,
                                                   const std::shared_ptr<ArtifactId> &input);

  /**
   * @brief generate transpose NCHW -> NHWC
   * @param name name of tensor containing transposed data
   * @param input_shape shape of @p input
   * @param input id of input tensor
   * @return Id of result tensor
   */
  std::shared_ptr<ArtifactId> genTransposeACLtoMIR(const std::string &name,
                                                   const mir::Shape &input_shape,
                                                   const std::shared_ptr<ArtifactId> &input);

  /**
   * @brief Generate DOM for PadStrideInfo object
   * @tparam Oper Class of operation with pad and stride properties
   * @param op Operation entity to generate variable for
   * @param prefix First part of generated variable name
   * @param block Code block where insert variable declaration
   * @return generated variable
   */
  template <typename Op>
  std::shared_ptr<ArtifactVariable> genPadStrideInfo(const Op &op, const std::string &prefix,
                                                     ArtifactBlock *block);

  template <typename Op>
  void genPooling(Op &op, const std::string &pooling_type, bool exclude_padding);

  /**
   * @brief The common part of the convolution and the depthwise convolution.
   */
  template <typename Op>
  void genConvolution(Op &op, const std::string &acl_func_name, const std::string &suffix);

  /**
   * @brief Generates different types of activation functions: ReLU, Tanh etc.
   * @param activation_name - names of activation functions used in ACL: RELU, TANH etc.
   * @param a - alpha parameter used by some activation functions: BOUNDED_RELU, LU_BOUNDED_RELU,
   *            LINEAR, TANH.
   * @param b - betha parameter used by some activation functions: LINEAR, LU_BOUNDED_RELU, TANH.
   */
  void genActivation(const mir::Operation &op, const std::string &activation_name, float a = 0,
                     float b = 0);

  /**
   * @brief            Used to generate a binary addition operation in handling of the elementwise.
   *
   * @param prefix   - the name (in the DOM) of operation called this method.
   * @param index          - the index of the call in the elementwise loop.
   * @param ir_shape - the shape of the operands in the IR.
   * @param in1      - the descriptor of the first operand in the DOM. Can be either original tensor
   *                   identifier in the input sequence or a variable storing the partial result of
   *                   applying the operation to the previous terms in the sequence.
   * @param in2      - the descriptor of the second operand in the DOM.
   * @param out      - the descriptor for storing the operation result. If it is not nullptr, it is
   *                   used to return the result. If it is nullptr (the default), a new tensor is
   *                   allocated in the DOM to return the result.
   * @return         - the DOM ID of the temporary variable storing the partial sum of the elements
   *                   to the left of and including the in2 term, or the operation out if in2 was
   *                   the last term in the sequence.
   */
  std::shared_ptr<ArtifactId> genAddition(const std::string &prefix, size_t index,
                                          const mir::Shape &ir_shape,
                                          const std::shared_ptr<ArtifactId> &in1,
                                          const std::shared_ptr<ArtifactId> &in2,
                                          std::shared_ptr<ArtifactId> out = nullptr);

  /**
   * @brief            Used to generate a binary multiplication operation in handling of the
   *                   elementwise. As there is currently no the CLArithmeticMultiplication in the
   *                   ACL library, in1 * in2 is emulated as: in1 / (1 / in2) with
   *                   CLArithmeticDivision.
   *
   * @param prefix   - the name (in the DOM) of operation called this method.
   * @param index    - the index of the call in the elementwise loop.
   * @param ir_shape - the shape of the operands in the IR.
   * @param in1      - the descriptor of the first operand in the DOM. Can be either original tensor
   *                   identifier in the input sequence or a variable storing the partial result of
   *                   applying the operation to the previous terms in the sequence.
   * @param in2      - the descriptor of the second operand in the DOM.
   * @param out      - the descriptor for storing the operation result. If it is not nullptr, it is
   *                   used to return the result. If it is nullptr (the default), a new tensor is
   *                   allocated in the DOM to return the result.
   * @return         - the DOM ID of the temporary variable storing the partial product of the
   *                   elements to the left of and including the in2 term, or the operation out if
   *                   in2 was the last term in the sequence.
   */
  std::shared_ptr<ArtifactId> genMultiplication(const std::string &prefix, size_t index,
                                                const mir::Shape &ir_shape,
                                                const std::shared_ptr<ArtifactId> &in1,
                                                const std::shared_ptr<ArtifactId> &in2,
                                                std::shared_ptr<ArtifactId> out = nullptr);

  /**
   * @brief Generates a unique name for the tensor.
   */
  std::string tensorName(const mir::Operation::Output *ir_tensor) const;

  /**
   * @brief Generates variables tensor shape in DOM.
   * @param block - DOM block where to create this shape: artifact constructor, inference function.
   * @param name - prefix used for generating the unique name for this shape.
   * @param shape - model IR shape for which we create a DOM analogue.
   * @return - a DOM identifier for the created shape.
   */
  template <typename T>
  std::shared_ptr<ArtifactId> genVectorInitializedVar(ArtifactBlock *block, const std::string &type,
                                                      const std::string &name,
                                                      const std::vector<T> &init);

  /**
   * @brief Generates a DOM tensor.
   * @param name - its name.
   * @param ir_shape - IR shape used to construct the tensor.
   * @param gen_accessor - whether to generate an accessor function for this tensor
   *        in the artifact class.
   * @return - a DOM identifier for the created tensor.
   */
  std::shared_ptr<ArtifactId> genTensor(const std::string &name, const mir::Shape &ir_shape,
                                        bool gen_accessor = false);

  /**
   * @brief Generates a DOM tensor.
   * @param ir_tensor - the ModelIR tensor.
   * @return - a DOM identifier for the created tensor.
   */
  std::shared_ptr<ArtifactId> genTensor(const mir::Operation::Output *ir_tensor);

  /**
   * @brief generate transposing operation, @p mir_perm contains dimensions in MIR order (batch has
   * index 0)
   * @param input id of input tensor
   * @param output id out output tensor
   * @param mir_perm new order of dimensions
   */
  void genTranspose(const std::shared_ptr<nnc::ArtifactId> &input,
                    const std::shared_ptr<nnc::ArtifactId> &output,
                    const std::vector<size_t> &mir_perm, bool allocate_at_inference);

  /**
   * @brief Generates accessors for the input/output tensors.
   * @param graph - the ModelIR graph.
   */
  void genNamed(mir::Graph *graph);

  /**
   * @brief Schedule a tensor serialization.
   * @param tensor_id - an artifact ID of the tensor.
   * @param ir_tensor - the IR source of the tensor.
   */
  void serializeTensor(const std::shared_ptr<ArtifactId> &tensor_id,
                       const mir::TensorVariant &ir_tensor);

  /**
   * @brief Serialize an IR tensor in a file.
   * @param tensor - tensor to serialize.
   */
  void serializeIRTensor(const mir::TensorVariant &tensor);

  /**
   * @brief Generate the deserialization calls in right places in the artifact.
   */
  void genDeserializations();

  /**
   * @brief Generate procedure calls for filling tensors with constant scalar values.
   */
  void genFillings();

  /**
   * @brief Store the tensor ID and its value for the successive generation (for uniform tensors).
   * @param tensor_id - ID of the tensor.
   * @param val - its value.
   */
  void fillTensor(const std::shared_ptr<ArtifactId> &tensor_id, const std::string &val);

  /**
   * @brief Schedule the tensor allocation in the artifact constructor.
   * @param tensor_id - ID of the scheduled tensor.
   */
  void addToPersistentTensors(const std::shared_ptr<ArtifactId> &tensor_id);

  /**
   * @brief Generate allocation of tensor
   * @param block Block to insert allocation in
   * @param tensor Id of tensor to allocate
   */
  std::shared_ptr<ArtifactFunctionCall>
  genTensorAllocation(ArtifactBlock *block, const std::shared_ptr<ArtifactId> &tensor);

  /**
   * @brief Generate deallocation of tensor
   * @param block Block to insert deallocation in
   * @param tensor Id of tensor to deallocate
   */
  std::shared_ptr<ArtifactFunctionCall>
  genTensorDeallocation(ArtifactBlock *block, const std::shared_ptr<ArtifactId> &tensor);

  /**
   * @brief Generate all the scheduled tensor allocations.
   */
  void genPersistentTensorAllocations();

  /**
   * @brief Generate the layer declaration and the configure() call.
   * @param layer_type - ACL layer type.
   * @param layer_name - name of the layer variable in the artifact.
   * @param config_params - input/output tensor names and the other configuration information.
   * @return - generated tensor ID.
   */
  std::shared_ptr<ArtifactId>
  genLayer(const std::string &layer_type, const std::string &layer_name,
           const std::list<std::shared_ptr<ArtifactExpr>> &config_params);

  /**
   * @brief Generate the layer run() call.
   * @param layer_id - layer ID.
   */
  void genLayerExecution(const std::shared_ptr<ArtifactId> &layer_id);

  /**
   * @brief All named tensors names.
   */
  std::set<std::string> _tensorNames;

  /**
   * @brief The stream for tensors serialization.
   */
  std::ostream &_parOut;

  /**
   * @brief The whole artifact module in the DOM.
   */
  ArtifactModule _module;

  /**
   * @brief The artifact class.
   */
  std::shared_ptr<ArtifactClass> _artifactClass;

  /**
   * @brief The artifact inference function.
   */
  std::shared_ptr<ArtifactClassFunction> _inferenceFunction;

  /**
   * @brief The constuctor block of DOM instructions.
   */
  ArtifactBlock *_constrBlock;

  /**
   * @brief The inference function block of DOM instruction.
   */
  ArtifactBlock *_infBlock;

  /**
   * @brief The variable describing the input stream for tensors deserialization.
   */
  std::shared_ptr<ArtifactVariable> _parInVar;

  /**
   * @brief The identifier to reference the previous variable.
   */
  std::shared_ptr<ArtifactId> _parIn;

  /**
   * @brief The CLScheduler class representation in the DOM.
   */
  std::shared_ptr<ArtifactId> _clScheduler;

  /**
   * @brief Tensors which need to be allocated at the artifact construction time.
   */
  std::list<std::shared_ptr<ArtifactId>> _persistent_tensors;

  /**
   * @brief Tensors which are serialized from the Model IR and need to be deserialized in the
   * artifact.
   */
  std::list<std::shared_ptr<ArtifactId>> _serializations;

  /**
   * @brief Tensors which must be filled with constant values and the corresponding values.
   */
  std::list<std::pair<std::shared_ptr<ArtifactId>, std::string>> _fillings;
};

} // namespace nnc

#endif //_NNC_ACL_CPP_OP_GENERATOR_H_
