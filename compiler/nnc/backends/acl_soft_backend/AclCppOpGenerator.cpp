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

#include "AclCppOpGenerator.h"
#include "backends/acl_soft_backend/AclCppException.h"
#include "mir/ShapeRange.h"
#include "mir/TensorUtil.h"
#include "mir/Tensor.h"

#include "mir/Operation.h"
#include "mir/OpDefs.h"

#include <algorithm>
#include <map>

namespace nnc
{

using namespace std;
using namespace mir;

AclCppOpGenerator::AclCppOpGenerator(const string &name, ostream &par_out)
  : _parOut(par_out), _module(name), _constrBlock(nullptr), _infBlock(nullptr),
    _clScheduler(AF::id("arm_compute::CLScheduler"))
{
}

const ArtifactModule &AclCppOpGenerator::generate(mir::Graph *g)
{
  // Including headers.
  _module.addHeaderSysInclude("fstream");
  _module.addHeaderInclude("arm_compute/core/Types.h");
  _module.addHeaderInclude("arm_compute/runtime/CL/CLFunctions.h");
  _module.addHeaderInclude("arm_compute/runtime/CL/CLScheduler.h");
  _module.addHeaderInclude("arm_compute/runtime/CL/CLBufferAllocator.h");
  _module.addHeaderInclude("arm_compute/runtime/BlobLifetimeManager.h");
  _module.addHeaderInclude("arm_compute/runtime/PoolManager.h");
  _module.addHeaderInclude("arm_compute/runtime/MemoryManagerOnDemand.h");

  // The general structure creation.
  _artifactClass = _module.createClass(_module.name());
  _constrBlock = _artifactClass->getConstrBlock();
  _inferenceFunction = _artifactClass->func(true, "void", "Inference");
  _infBlock = _inferenceFunction->getBlock();

  // Input parameter stream preparation.
  _parInVar = _artifactClass->var(false, "std::ifstream", "_parIn");
  _parIn = _parInVar->use();
  string par_file_name = _module.name() + ".par";
  _constrBlock->call(
    "open",
    {AF::lit("\"" + par_file_name + "\""), AF::lit("std::ios_base::in | std::ios_base::binary")},
    _parIn);
  auto file_fail = _constrBlock->ifCond(AF::call("fail", {}, _parIn));
  auto file_fail_block = file_fail->getBlock();
  file_fail_block->addStatement(
    AF::lit("throw std::string(\"Failed to open file: " + par_file_name + " for reading\")"));

  // Traverse the computational graph.
  g->accept(this);

  // Generate all the deferred entities.
  genNamed(g);
  genPersistentTensorAllocations();
  genDeserializations();
  genFillings();

  // Make sure all the OpenCL jobs are done executing:
  _infBlock->call("sync", {}, AF::call("get", {}, _clScheduler, ArtifactCallType::scope));

  return _module;
}

void AclCppOpGenerator::visit(ops::ConcatOp &op)
{
  const auto &ir_inputs = op.getInputs();
  const auto *ir_output = op.getOutput(0);

  static const char *axis_names[] = {
    "arm_compute::DataLayoutDimension::BATCHES", "arm_compute::DataLayoutDimension::CHANNEL",
    "arm_compute::DataLayoutDimension::HEIGHT", "arm_compute::DataLayoutDimension::WIDTH"};

  int axis = op.getAxis();
  assert(axis >= 0 && axis < static_cast<int>(sizeof(axis_names) / sizeof(axis_names[0])) &&
         "axis outside this range is not supported in ACL");
  const char *axis_name = axis_names[axis];

  auto out = genTensor(ir_output);
  auto prefix = out->name() + "_concatenate_layer";
  auto inputs_var = _constrBlock->var("std::vector<arm_compute::ICLTensor*>", prefix + "_inputs");
  auto inputs = inputs_var->use();

  for (const Operation::Output *ir_input : ir_inputs)
    _constrBlock->call("push_back", {AF::ref(AF::id(tensorName(ir_input)))}, inputs);

  auto layer =
    genLayer("arm_compute::CLConcatenateLayer", prefix, {inputs, AF::ref(out), AF::lit(axis_name)});

  addToPersistentTensors(out);
  genLayerExecution(layer);
}

void AclCppOpGenerator::visit(ops::Conv2DOp &op)
{
  assert(op.getNumGroups() == 1);
  genConvolution(op, "arm_compute::CLConvolutionLayer", "_convolution_layer");
}

void AclCppOpGenerator::visit(ops::DepthwiseConv2DOp &op)
{
  genConvolution(op, "arm_compute::CLDepthwiseConvolutionLayer", "_depthwise_convolution_layer");
}

void AclCppOpGenerator::visit(ops::SoftmaxOp &op)
{
  assert(op.getNumInputs() == 1);
  const auto *ir_input = op.getInput(0);
  const auto *ir_output = op.getOutput(0);

  auto in = AF::id(tensorName(ir_input));

  int rank = ir_output->getShape().rank();
  // CLPermute does not support all kinds of permutations now.
  // rank can be more than 2 in our models, so we can not use CLTranspose.
  // This means we can support tensors with no more then one axis > 1.
  int axis = op.getAxis();
  assert(axis == rank - 1);
  int nof_long_axes = 0;

  for (int i = 0; i < rank; ++i)
  {
    if (ir_output->getShape().dim(i) > 1)
      ++nof_long_axes;
  }

  // TODO: Consider how to support Softmax on more general inputs.
  if (nof_long_axes > 1)
    throw AclCppException("Unsupported Softmax operation with several dimensions greater than 1");

  // Create the output tensor.
  shared_ptr<ArtifactId> output = genTensor(ir_output);
  auto layer_name_prefix = output->name();

  if (axis == 0)
  {
    // Simple version: do not need pre and post reshapes.
    // Apply the softmax operation.
    auto sm = genLayer("arm_compute::CLSoftmaxLayer", layer_name_prefix + "_softmax_layer",
                       {AF::ref(in), AF::ref(output)});
    addToPersistentTensors(output);
    genLayerExecution(sm);
  }
  else
  {
    // TODO refactor this code, it works only with 1 batch

    // Need to reshape before the Softmax application and after it.
    // Then we need two tensors for intermediate results. This is because we do a couple of
    // auxiliary
    // reshapes: one to transform the input tensor to a unidimensional tensor and the second to
    // transorm the result of the softmax operation back to the original form.
    Shape sm_shape(ir_output->getShape());

    std::swap(sm_shape.dim(axis), sm_shape.dim(-1));

    auto tmp = genTensor(layer_name_prefix + "_tmp", sm_shape);
    auto tmp2 = genTensor(layer_name_prefix + "_tmp2", sm_shape);

    // Do the input permutation.
    auto transp1 = genLayer("arm_compute::CLReshapeLayer", layer_name_prefix + "_transp_layer1",
                            {AF::ref(in), AF::ref(tmp)});
    addToPersistentTensors(tmp);
    genLayerExecution(transp1);

    // Apply the softmax operaion.
    auto sm = genLayer("arm_compute::CLSoftmaxLayer", layer_name_prefix + "_softmax_layer",
                       {AF::ref(tmp), AF::ref(tmp2)});
    addToPersistentTensors(tmp2);
    genLayerExecution(sm);

    // Reshape the output to the original form.
    auto transp2 = genLayer("arm_compute::CLReshapeLayer", layer_name_prefix + "_transp_layer2",
                            {AF::ref(tmp2), AF::ref(output)});
    addToPersistentTensors(output);
    genLayerExecution(transp2);
  }
}

template <typename Op>
shared_ptr<ArtifactVariable> AclCppOpGenerator::genPadStrideInfo(const Op &op, const string &prefix,
                                                                 ArtifactBlock *block)
{
  using AF = ArtifactFactory;

  const Shape strides(op.getStrides());
  assert(strides.rank() == 2);
  auto &padding_before = op.getPaddingBefore();
  auto &padding_after = op.getPaddingAfter();

  string type_name = "arm_compute::PadStrideInfo";

  string var_name = prefix + "_pad_stride_info";

  list<std::shared_ptr<ArtifactExpr>> var_init_params = {
    AF::lit(to_string(strides.dim(1))),
    AF::lit(to_string(strides.dim(0))),
    AF::lit(to_string(padding_before.at(1))),
    AF::lit(to_string(padding_after.at(1))),
    AF::lit(to_string(padding_before.at(0))),
    AF::lit(to_string(padding_after.at(0))),
    AF::lit("arm_compute::DimensionRoundingType::FLOOR")};

  auto pad_stride_info_var = block->var(type_name, var_name, {}, var_init_params);

  return pad_stride_info_var;
}

shared_ptr<ArtifactId> AclCppOpGenerator::genTransposeMIRtoACL(const string &name,
                                                               const Shape &input_shape,
                                                               const shared_ptr<ArtifactId> &input)
{
  Shape transposed_shape = transposeShape<0, 3, 1, 2>(input_shape);
  shared_ptr<ArtifactId> transposed_id = genTensor(name, transposed_shape, false);
  const bool allocate_at_inference = true;
  genTranspose(input, transposed_id, {0, 3, 1, 2}, allocate_at_inference);
  return transposed_id;
}

shared_ptr<ArtifactId> AclCppOpGenerator::genTransposeACLtoMIR(const string &name,
                                                               const Shape &input_shape,
                                                               const shared_ptr<ArtifactId> &input)
{
  Shape transposed_shape = transposeShape<0, 2, 3, 1>(input_shape);
  shared_ptr<ArtifactId> transposed_id = genTensor(name, transposed_shape, false);
  const bool allocate_at_inference = false;
  genTranspose(input, transposed_id, {0, 2, 3, 1}, allocate_at_inference);
  return transposed_id;
}

void AclCppOpGenerator::visit(ops::AvgPool2DOp &op)
{
  genPooling(op, "arm_compute::PoolingType::AVG", !op.getIncludePad());
}

void AclCppOpGenerator::visit(ops::MaxPool2DOp &op)
{
  // The value of 'exclude_padding' does not really matter for MAX pooling.
  genPooling(op, "arm_compute::PoolingType::MAX", false);
}

void AclCppOpGenerator::visit(ops::FullyConnectedOp &op)
{
  assert(op.getNumInputs() == 2);
  const auto *ir_input = op.getInput(0);
  const auto *ir_weights = op.getInput(1);
  const auto *ir_output = op.getOutput(0);

  auto ir_weights_op = dynamic_cast<const mir::ops::ConstantOp *>(ir_weights->getNode());
  if (ir_weights_op == nullptr)
    throw AclCppException("Unsupported operation type");

  const TensorVariant ir_weights_tensor = transposeTensor<1, 0>(ir_weights_op->getValue());
  const Shape &ir_weights_shape = ir_weights_tensor.getShape();

  // Get the input node tensor id in the DOM.
  auto in = AF::id(tensorName(ir_input));

  // Create the output tensor in the DOM.
  if (ir_output->getShape().rank() != 2)
    throw AclCppException("Unsupported number of dimensions in fc layer");
  auto out = genTensor(ir_output);
  string operation_name = out->name() + "_fully_connected_layer";

  // Create the weights tensor in the DOM and use its id.
  auto weights = genTensor(operation_name + "_weights", ir_weights_shape);

  // Instantiate the CLFullyConnectedLayer object.
  auto layer = genLayer("arm_compute::CLFullyConnectedLayer", operation_name,
                        {AF::ref(in), AF::ref(weights), AF::lit("nullptr"), AF::ref(out)});

  addToPersistentTensors(weights);
  // Serialize the weights tensor and generate the function to deserialize it in the artifact.
  serializeTensor(weights, ir_weights_tensor);
  addToPersistentTensors(out);
  genLayerExecution(layer);
}

void AclCppOpGenerator::visit(ops::CappedReluOp &op)
{
  genActivation(op, "LU_BOUNDED_RELU", op.getCap());
}

void AclCppOpGenerator::visit(ops::InputOp &op)
{
  shared_ptr<ArtifactId> tensor;
  tensor = genTensor(op.getOutput(0));
  addToPersistentTensors(tensor);
}

// FIXME: temporary solution
static bool shouldSerializeConstant(const ops::ConstantOp &op)
{
  // Operations from 'self_serializing_ops_to_inputs' serializing tensors with appropriate index
  // themselves,
  // so we don't serialize them here, also we don't serialize tensors from dangling ConstantOp
  static std::map<Operation::Type, std::size_t> self_serializing_ops_to_inputs{
    {Operation::Type::conv2D, 1}, {Operation::Type::fullyConnected, 1}};

  for (Operation::Use use : op.getOutput(0)->getUses())
  {
    auto self_serializing_op_it = self_serializing_ops_to_inputs.find(use.getNode()->getType());
    // Serialize if next_node type not from 'self_serializing_ops_to_inputs'
    if (self_serializing_op_it == self_serializing_ops_to_inputs.end())
      return true;

    // If next_node has current ConstantOp as it's previous node, but not with appropriate index -
    // serialize current ConstantOp
    if (self_serializing_op_it->second != use.getIndex())
      return true;
  }

  return false;
}

void AclCppOpGenerator::visit(ops::ConstantOp &op)
{
  if (shouldSerializeConstant(op))
  {
    TensorVariant data = op.getValue();
    shared_ptr<ArtifactId> out = genTensor(op.getOutput(0));
    addToPersistentTensors(out);
    serializeTensor(out, data);
  }
}

void AclCppOpGenerator::visit(ops::ReluOp &op) { genActivation(op, "RELU"); }

void AclCppOpGenerator::visit(ops::ReshapeOp &op)
{
  assert(op.getNumInputs() == 1);
  const auto *ir_input = op.getInput(0);
  const auto *ir_output = op.getOutput(0);

  // Get the id of the input tensor in the generated artifact.
  auto in = AF::id(tensorName(ir_input));

  // Create the output tensor in the DOM and return its id.
  const Shape &out_shape = ir_output->getShape();

  // This check confirms that we can "safely" reshape data
  // The only safe configuration of output shape is (1...1, N, 1 ... 1)
  bool found_non_one = false;
  for (int32_t i = 0; i < out_shape.rank(); ++i)
  {
    if (out_shape.dim(i) != 1)
    {
      if (found_non_one)
        throw AclCppException("Unsupported result of reshape");
      found_non_one = true;
    }
  }

  shared_ptr<ArtifactId> out = genTensor(ir_output);

  // Create an instance of the CLReshapeLayer class as a member of the artifact class.
  auto layer = genLayer("arm_compute::CLReshapeLayer", out->name() + "_reshape_layer",
                        {AF::ref(in), AF::ref(out)});
  addToPersistentTensors(out);
  genLayerExecution(layer);
}

void AclCppOpGenerator::visit(mir::ops::SliceOp & /*op*/)
{
  throw AclCppException("Unimplemented operation: SliceOp");
}

void AclCppOpGenerator::visit(ops::TanhOp &op) { genActivation(op, "TANH"); }

void AclCppOpGenerator::visit(ops::DeConv2DOp &op)
{
  genConvolution(op, "arm_compute::CLDeconvolutionLayer", "_deconvolution_layer");
}

void AclCppOpGenerator::visit(ops::EluOp & /*op*/)
{
  throw AclCppException("EluOp not supported by the ACL library yet.");
}

void AclCppOpGenerator::visit(ops::PadOp &op)
{
  assert(op.getNumInputs() == 1);
  const auto *ir_input = op.getInput(0);
  const auto *ir_output = op.getOutput(0);

  // Get the id of the input tensor.
  auto input = AF::id(tensorName(ir_input));

  // Create the output tensor in the DOM
  auto out = genTensor(ir_output);
  addToPersistentTensors(out);

  // Generate PadLayer params
  auto prefix = out->name() + "_pad_layer";
  auto pad_list_decl = _constrBlock->var("arm_compute::PaddingList", prefix + "_pads");
  auto pad_list = pad_list_decl->use();
  const auto &padding_before = op.getPaddingBefore();
  const auto &padding_after = op.getPaddingAfter();
  for (int i = 0; i < ir_input->getShape().rank(); ++i)
  {
    auto pad_var = _constrBlock->var(
      "arm_compute::PaddingInfo", prefix + "_pad_" + to_string(i), {},
      {AF::lit(to_string(padding_before[i])), AF::lit(to_string(padding_after[i]))});
    auto pad = pad_var->use();
    _constrBlock->call("push_back", {pad}, pad_list);
  }

  // Generate PadLayer
  // FIXME Set up the `constant_value` parameter.
  assert(op.getPaddingValue() == 0.0f);
  auto layer =
    genLayer("arm_compute::CLPadLayer", prefix, {AF::ref(input), AF::ref(out), pad_list});
  genLayerExecution(layer);
}

template <typename Op>
void AclCppOpGenerator::genPooling(Op &op, const std::string &pooling_type, bool exclude_padding)
{
  assert(op.getNumInputs() == 1);
  const auto *ir_input = op.getInput(0);
  const auto *ir_output = op.getOutput(0);

  string in_name = tensorName(ir_input);
  auto in_id = AF::id(in_name);

  const string output_tensor_name = tensorName(ir_output);

  // Transpose data from MIR format to format compatible with ACL
  const string transposed_input_name = output_tensor_name + "transposed_input";
  shared_ptr<ArtifactId> transposed_input =
    genTransposeMIRtoACL(transposed_input_name, ir_input->getShape(), in_id);

  const string layer_name = output_tensor_name + "_pooling_layer";

  shared_ptr<ArtifactVariable> pad_stride_info_var = genPadStrideInfo(op, layer_name, _constrBlock);

  shared_ptr<ArtifactId> pad_stride_info = pad_stride_info_var->use();

  // Create kernel window info
  shared_ptr<ArtifactVariable> kernel_window_var = _constrBlock->var(
    "arm_compute::Size2D", layer_name + "_kernel_window", {},
    {AF::lit(to_string(op.getWindowSize()[1])), AF::lit(to_string(op.getWindowSize()[0]))});
  shared_ptr<ArtifactId> kernel_window = kernel_window_var->use();

  // Create pooling info: pooling type, kernel info, strides, etc
  shared_ptr<ArtifactVariable> pooling_info_var =
    _constrBlock->var("arm_compute::PoolingLayerInfo", layer_name + "_pooling_info", {},
                      {AF::lit(pooling_type), kernel_window, pad_stride_info,
                       AF::lit(exclude_padding ? "true" : "false")});
  shared_ptr<ArtifactId> pooling_info = pooling_info_var->use();

  // Generate auxiliary tensor to hold transposed output of pool in NCHW format
  Shape transposed_output_shape = transposeShape<0, 3, 1, 2>(ir_output->getShape());
  shared_ptr<ArtifactId> transposed_output =
    genTensor(layer_name + "_out_transpose", transposed_output_shape);

  // Actual layer creation
  shared_ptr<ArtifactId> layer =
    genLayer("arm_compute::CLPoolingLayer", layer_name,
             {AF::ref(transposed_input), AF::ref(transposed_output), pooling_info});
  genTensorAllocation(_infBlock, transposed_output);
  genLayerExecution(layer);

  shared_ptr<ArtifactId> output =
    genTransposeACLtoMIR(output_tensor_name, transposed_output_shape, transposed_output);

  genTensorDeallocation(_infBlock, transposed_input);
  genTensorDeallocation(_infBlock, transposed_output);
}

template <typename Op>
void AclCppOpGenerator::genConvolution(Op &op, const string &acl_func_name, const string &suffix)
{
  const auto *ir_input = op.getInput(0);
  const auto *ir_weights = op.getInput(1);
  const auto *ir_output = op.getOutput(0);

  auto ir_weights_op = dynamic_cast<const ops::ConstantOp *>(ir_weights->getNode());
  if (ir_weights_op == nullptr)
    throw AclCppException("Unsupported operation type");

  auto ir_weights_tensor = ir_weights_op->getValue();
  if (op.getType() == Operation::Type::conv2D)
  {
    // [Co, Hk, Wk, Ci] -> [Co, Ci, Hk, Wk].
    ir_weights_tensor = transposeTensor<0, 3, 1, 2>(ir_weights_tensor);
  }
  else
  {
    ir_weights_tensor = transposeTensor<3, 2, 0, 1>(ir_weights_tensor);
  }

  const Shape &ir_weights_shape = ir_weights_tensor.getShape();

  // get output tensor name that is used as base for other names
  const string output_tensor_name = tensorName(ir_output);

  // Get the identifier of the input tensor in the DOM.
  auto input = AF::id(tensorName(ir_input));

  // Generate auxiliary tensor to hold transposed input of convolution in NCHW format
  shared_ptr<ArtifactId> transposed_input =
    genTransposeMIRtoACL(output_tensor_name + "_transposed_input", ir_input->getShape(), input);

  // Create the transposed output tensor in the DOM.
  const string transposed_output_name = output_tensor_name + "_transposed_output";
  Shape transposed_output_shape = transposeShape<0, 3, 1, 2>(ir_output->getShape());
  shared_ptr<ArtifactId> transposed_output =
    genTensor(transposed_output_name, transposed_output_shape);

  string operation_name = output_tensor_name + suffix;

  // Generate a tensor for weights (kernel) in the DOM.
  auto weights = genTensor(operation_name + "_weights", ir_weights_shape);

  // Create a local variable of type PadStrideInfo in the artifact constructor:
  // PadStrideInfo pad_stride_info(stride_x, stride_y, pad_x, pad_y);
  auto pad_stride_info_var = genPadStrideInfo(op, operation_name, _constrBlock);

  auto pad_stride_info = pad_stride_info_var->use();

  // The parameter for the conv_layer.config(&in, &weights, nullptr, &out, pad_stride_info)
  // function call.
  list<shared_ptr<ArtifactExpr>> config_params{AF::ref(transposed_input), AF::ref(weights),
                                               AF::lit("nullptr"), AF::ref(transposed_output),
                                               pad_stride_info};

  // Add to additional parameters for deconvolution.
  if (op.getType() == Operation::Type::deConv2D)
  {
    config_params.push_back(AF::lit("0"));
    config_params.push_back(AF::lit("0"));
  }

  // Create the convolution (/depthwise convolution/deconvolution) layer class instance.
  shared_ptr<ArtifactId> layer = genLayer(acl_func_name, operation_name, config_params);

  addToPersistentTensors(weights);
  // Save the IR weights tensor to later read this in the artifact.
  serializeTensor(weights, ir_weights_tensor);
  genTensorAllocation(_infBlock, transposed_output);
  genLayerExecution(layer);

  // Generate auxiliar tensor to hold transposed output of convolution in NHWC format
  shared_ptr<ArtifactId> output =
    genTransposeACLtoMIR(output_tensor_name, transposed_output_shape, transposed_output);

  genTensorDeallocation(_infBlock, transposed_input);
  genTensorDeallocation(_infBlock, transposed_output);
}

void AclCppOpGenerator::genActivation(const Operation &op, const std::string &activation_name,
                                      float a, float b)
{
  assert(op.getNumInputs() == 1);
  const auto *ir_input = op.getInput(0);
  const auto *ir_output = op.getOutput(0);

  // Get the id of the input tensor.
  auto in = AF::id(tensorName(ir_input));

  // Create the output tensor in the DOM and return its id.
  shared_ptr<ArtifactId> output = genTensor(ir_output);

  auto prefix = output->name() + "_activation_layer";

  // Create an instance of the ActivationLayerInfo class as a local variable in the artifact
  // constructor. This instance profide information about the concrete activation function,
  // like: ReLU, Tanh etc and two optional parameter (alpha and betha) needed by some activations.
  auto activation_info_var = _constrBlock->var(
    "arm_compute::ActivationLayerInfo", prefix + "_activation_info", {},
    {AF::lit("arm_compute::ActivationLayerInfo::ActivationFunction::" + activation_name),
     AF::lit(to_string(a)), AF::lit(to_string(b))});
  auto activation_info = activation_info_var->use();

  // Create an instance of the CLActivationLayer class as a member of the artifact class.
  auto layer = genLayer("arm_compute::CLActivationLayer", prefix,
                        {AF::ref(in), AF::ref(output), activation_info});
  addToPersistentTensors(output);
  genLayerExecution(layer);
}

shared_ptr<ArtifactId> AclCppOpGenerator::genAddition(const string &prefix, size_t index,
                                                      const Shape &ir_shape,
                                                      const std::shared_ptr<ArtifactId> &in1,
                                                      const std::shared_ptr<ArtifactId> &in2,
                                                      std::shared_ptr<ArtifactId> out)
{
  string operation_name = prefix + "_" + to_string(index);
  // Create the output tensor in the DOM or reuse the out, if it is not nullptr - that is for the
  // last element in the handled sequence.
  if (!out)
    out = genTensor(operation_name, ir_shape);

  // Create an instance of the CLActivationLayer class as a member of the artifact class.
  auto arithmetic_add_layer_var = _artifactClass->var(false, "arm_compute::CLArithmeticAddition",
                                                      operation_name + "_arithmetic_add_layer");
  auto arithmetic_add_layer = arithmetic_add_layer_var->use();

  // Generate the call: arithmetic_add_layer.configure(&in1, &in2, &out);
  _constrBlock->call(
    "configure",
    {AF::ref(in1), AF::ref(in2), AF::ref(out), AF::lit("arm_compute::ConvertPolicy::WRAP")},
    arithmetic_add_layer);

  // Generate the call: arithmetic_add_layer.run();
  _infBlock->call("run", {}, arithmetic_add_layer);
  return out;
}

shared_ptr<ArtifactId> AclCppOpGenerator::genMultiplication(const string &prefix, size_t index,
                                                            const Shape &ir_shape,
                                                            const shared_ptr<ArtifactId> &in1,
                                                            const shared_ptr<ArtifactId> &in2,
                                                            shared_ptr<ArtifactId> out)
{
  string operation_name = prefix + "_" + to_string(index);

  // Create the output tensor in the DOM or reuse the out, if it is not nullptr - that is for the
  // last element in the handled sequence.
  if (!out)
    out = genTensor(operation_name, ir_shape);

  // Create a unit tensor with the rank = ir.shape.rank() and having all dimensions = 1. It is
  // possible to use such a tensor in the operation because of the broadcasting support for the
  // input tensors in the CLArithmeticDivision operation.
  Shape ir_unit_shape(ir_shape.rank());

  for (int i = 0; i < ir_unit_shape.rank(); ++i)
    ir_unit_shape.dim(i) = 1;

  // Create a unit tensor in the DOM.
  auto unit = genTensor(operation_name + "_unit", ir_unit_shape);
  addToPersistentTensors(unit);

  // Fill the unit tensor with the 1 value.
  fillTensor(unit, "1");

  // Create a tmp tensor in the DOM to store the result of 1 / in2.
  auto tmp = genTensor(operation_name + "_tmp", ir_shape);
  genTensorAllocation(_infBlock, tmp);

  // Create an instance of the CLArithmeticDivision class as a member of the artifact class.
  auto arithmetic_div_layer_var1 = _artifactClass->var(false, "arm_compute::CLArithmeticDivision",
                                                       operation_name + "_arithmetic_div_layer_1");
  auto arithmetic_div_layer1 = arithmetic_div_layer_var1->use();

  // Generate the call: arithmetic_div_layer1.configure(&unit, &in2, &tmp);
  _constrBlock->call("configure", {AF::ref(unit), AF::ref(in2), AF::ref(tmp)},
                     arithmetic_div_layer1);

  // Generate the call: arithmetic_div_layer1.run();
  _infBlock->call("run", {}, arithmetic_div_layer1);

  // Create an instance of the CLArithmeticDivision class as a member of the artifact class.
  auto arithmetic_div_layer_var2 = _artifactClass->var(false, "arm_compute::CLArithmeticDivision",
                                                       operation_name + "_arithmetic_div_layer_2");
  auto arithmetic_div_layer2 = arithmetic_div_layer_var2->use();

  // Generate the call: arithmetic_div_layer2.configure(&in1, &tmp, &out);
  _constrBlock->call("configure", {AF::ref(in1), AF::ref(tmp), AF::ref(out)},
                     arithmetic_div_layer2);

  // Generate the call: arithmetic_div_layer2.run();
  _infBlock->call("run", {}, arithmetic_div_layer2);

  genTensorDeallocation(_infBlock, tmp);

  return out;
}

string AclCppOpGenerator::tensorName(const Operation::Output *ir_tensor) const
{
  string tensor_name = ir_tensor->getName();

  if (!tensor_name.empty())
  {
    tensor_name = "_" + tensor_name;
    replace_if(
      tensor_name.begin(), tensor_name.end(), [](char c) { return std::isalnum(c) == 0; }, '_');
  }
  else
  {
    assert(ir_tensor->getNode()->getNumOutputs() == 1);
    tensor_name = "tensor_" + to_string(ir_tensor->getNode()->getId());
  }

  return tensor_name;
}

template <typename T>
std::shared_ptr<ArtifactId>
AclCppOpGenerator::genVectorInitializedVar(ArtifactBlock *block, const string &type,
                                           const string &name, const vector<T> &init)
{
  list<shared_ptr<ArtifactExpr>> dims;

  for (const auto &v : init)
    dims.push_back(AF::lit(to_string(v)));

  auto shape_var = block->var(type, name, {}, dims);
  auto shape_id = shape_var->use();
  return shape_id;
}

shared_ptr<ArtifactId> AclCppOpGenerator::genTensor(const string &name, const Shape &ir_shape,
                                                    bool gen_accessor)
{
  auto id = AF::id(name);

  if (_tensorNames.insert(name).second)
  {
    _artifactClass->var(false, "arm_compute::CLTensor", name);
    vector<int32_t> shape_vectorized;

    // create vector of initializers from Shape
    shape_vectorized.reserve(ir_shape.rank());
    for (int i = 0; i < ir_shape.rank(); ++i)
      shape_vectorized.push_back(ir_shape.dim(-i - 1));

    const char *type_name = "arm_compute::TensorShape";
    shared_ptr<ArtifactId> shape =
      genVectorInitializedVar(_constrBlock, type_name, name + "_shape", shape_vectorized);
    _constrBlock->call("initializeTensor", {id, shape});

    if (gen_accessor)
    {
      auto f = _artifactClass->func(true, "arm_compute::CLTensor&", "get" + name);
      auto b = f->getBlock();
      b->ret(id);
    }
  }

  return id;
}

shared_ptr<ArtifactId> AclCppOpGenerator::genTensor(const Operation::Output *ir_tensor)
{
  return genTensor(tensorName(ir_tensor), ir_tensor->getShape(), !ir_tensor->getName().empty());
}

void AclCppOpGenerator::genNamed(Graph *graph)
{
  const auto &inputs = graph->getInputs();
  if (inputs.size() == 1)
  {
    const auto *input_op = inputs[0];
    auto f = _artifactClass->func(true, "arm_compute::CLTensor&", "getInput");
    auto b = f->getBlock();
    auto id = AF::id(tensorName(input_op->getOutput(0)));
    b->ret(id);
  }

  const auto &outputs = graph->getOutputs();
  if (outputs.size() == 1)
  {
    const auto *output_op = outputs[0];
    auto f = _artifactClass->func(true, "arm_compute::CLTensor&", "getOutput");
    auto b = f->getBlock();
    auto id = AF::id(tensorName(output_op->getInput(0)));
    b->ret(id);
  }
}

void AclCppOpGenerator::serializeTensor(const shared_ptr<ArtifactId> &tensor_id,
                                        const TensorVariant &ir_tensor)
{
  serializeIRTensor(ir_tensor);
  _serializations.push_back(tensor_id);
}

void AclCppOpGenerator::serializeIRTensor(const TensorVariant &tensor)
{
  const Shape &shape = tensor.getShape();
  Index coords;
  coords.resize(shape.rank());
  Index dimensions;
  dimensions.resize(shape.rank());

  for (int i = 0; i < shape.rank(); ++i)
  {
    coords.at(i) = 0;
    dimensions.at(i) = shape.dim(i);
  }

  size_t data_size = tensor.getElementSize() * tensor.getShape().numElements();
  _parOut.write(tensor.atOffset(0), data_size);
}

void AclCppOpGenerator::genDeserializations()
{
  for (auto &tensor : _serializations)
    _constrBlock->call("deserializeTensor", {_parIn, tensor});
}

void AclCppOpGenerator::genFillings()
{
  for (auto f : _fillings)
    _constrBlock->call("fillTensor", {f.first, AF::lit(f.second)});
}

void AclCppOpGenerator::fillTensor(const shared_ptr<ArtifactId> &tensor_id, const string &val)
{
  _fillings.emplace_back(make_pair(tensor_id, val));
}

void AclCppOpGenerator::visit(ops::SqueezeOp & /*op*/)
{
  throw AclCppException("Unimplemented operation: Squeeze");
}

void AclCppOpGenerator::visit(ops::SqrtOp & /*op*/)
{
  throw AclCppException("Unimplemented operation: Sqrt");
}

void AclCppOpGenerator::addToPersistentTensors(const std::shared_ptr<ArtifactId> &tensor_id)
{
  _persistent_tensors.push_back(tensor_id);
}

shared_ptr<ArtifactFunctionCall>
AclCppOpGenerator::genTensorAllocation(ArtifactBlock *block, const shared_ptr<ArtifactId> &tensor)
{
  return block->call("allocate", {}, AF::call("allocator", {}, tensor), ArtifactCallType::ref);
}

shared_ptr<ArtifactFunctionCall>
AclCppOpGenerator::genTensorDeallocation(ArtifactBlock *block, const shared_ptr<ArtifactId> &tensor)
{
  return block->call("free", {}, AF::call("allocator", {}, tensor), ArtifactCallType::ref);
}

void AclCppOpGenerator::genPersistentTensorAllocations()
{
  for (auto &tensor : _persistent_tensors)
    genTensorAllocation(_constrBlock, tensor);
}

shared_ptr<ArtifactId>
AclCppOpGenerator::genLayer(const string &layer_type, const string &layer_name,
                            const list<shared_ptr<ArtifactExpr>> &config_params)
{
  auto layer_var = _artifactClass->var(false, layer_type, layer_name);
  auto layer = layer_var->use();
  _constrBlock->call("configure", config_params, layer);
  return layer;
}

void AclCppOpGenerator::genLayerExecution(const shared_ptr<ArtifactId> &layer_id)
{
  _infBlock->call("run", {}, layer_id);
}

void AclCppOpGenerator::visit(mir::ops::ResizeOp & /*op*/)
{
  throw AclCppException("Unimplemented operation: Resize");
}

void AclCppOpGenerator::genTranspose(const std::shared_ptr<nnc::ArtifactId> &input,
                                     const std::shared_ptr<nnc::ArtifactId> &output,
                                     const std::vector<size_t> &mir_perm,
                                     bool allocate_at_inference)
{

  // acl 18.8 opencl implementation supports only 3 types of permutation:
  // in mir (0, 3, 1, 2),  in acl(axes are in reverse order) (1, 2, 0)
  // in mir (0, 2, 3, 1),  in acl (2, 0, 1)
  // in mir (2, 3, 1, 0),  in acl (3, 2, 0, 1)
  // so here we try to transform mir transpose into one acl supports

  const string &out_name = output->name();
  vector<size_t> acl_perm;

  if (mir_perm == vector<size_t>{0, 3, 1, 2})
    acl_perm = {1, 2, 0};
  else if (mir_perm == vector<size_t>{0, 2, 3, 1})
    acl_perm = {2, 0, 1};
  else if (mir_perm == vector<size_t>{2, 3, 1, 0})
    acl_perm = {3, 2, 0, 1};
  else
    throw AclCppException("Unsupported transpose sequence in operation " + out_name);

  // Create operation parameter containing permutation vector
  shared_ptr<ArtifactId> perm_vector = genVectorInitializedVar(
    _constrBlock, "arm_compute::PermutationVector", out_name + "_perm_param", acl_perm);

  // Instantiate the CLPermute object.
  string layer_name = out_name + "_transpose_layer";
  list<shared_ptr<ArtifactExpr>> arguments = {AF::ref(input), AF::ref(output), perm_vector};
  auto layer = genLayer("arm_compute::CLPermute", layer_name, arguments);
  if (allocate_at_inference)
    genTensorAllocation(_infBlock, output);
  else
    addToPersistentTensors(output);
  genLayerExecution(layer);
}

void AclCppOpGenerator::visit(mir::ops::TransposeOp &op)
{
  assert(op.getNumInputs() == 1);
  const auto *ir_input = op.getInput(0);
  const auto *ir_output = op.getOutput(0);

  // Get the input node tensor id in the DOM.
  shared_ptr<ArtifactId> input = AF::id(tensorName(ir_input));
  const vector<size_t> &mir_axis_order = op.getAxisOrder();

  // Create the output tensor in the DOM.
  if (ir_output->getShape().rank() != 4)
    throw AclCppException("Unsupported number of dimensions in transpose operation");
  // TODO replace transpose shape
  shared_ptr<ArtifactId> output = genTensor(ir_output);

  // Actual generation of operation and related stuff
  genTranspose(input, output, mir_axis_order, false);
}

void AclCppOpGenerator::visit(mir::ops::GatherOp & /*op*/)
{
  throw AclCppException("Unimplemented operation: GatherOp");
}

void AclCppOpGenerator::visit(ops::SigmoidOp &op) { genActivation(op, "LOGISTIC"); }

void AclCppOpGenerator::visit(mir::ops::LeakyReluOp &op)
{
  genActivation(op, "LEAKY_RELU", op.getAlpha());
}

void AclCppOpGenerator::visit(mir::ops::OutputOp & /*op*/)
{
  // No-op.
}

void AclCppOpGenerator::visit(mir::ops::AddOp &op)
{
  assert(op.getNumInputs() == 2);
  const auto *ir_lhs = op.getInput(0);
  const auto *ir_rhs = op.getInput(1);
  const auto *ir_output = op.getOutput(0);

  // Create the output tensor in the DOM and obtain its identifier.
  auto out = genTensor(ir_output);
  addToPersistentTensors(out);

  // Get the identifiers of the input tensors in the DOM.
  auto lhs = AF::id(tensorName(ir_lhs));
  auto rhs = AF::id(tensorName(ir_rhs));

  genAddition(out->name() + "_" + "addition", 0, ir_rhs->getShape(), lhs, rhs, out);
}

void AclCppOpGenerator::visit(mir::ops::DivOp &) { throw AclCppException("NYI"); }

void AclCppOpGenerator::visit(mir::ops::MaxOp &) { throw AclCppException("NYI"); }

void AclCppOpGenerator::visit(mir::ops::MulOp &op)
{
  assert(op.getNumInputs() == 2);
  const auto *ir_lhs = op.getInput(0);
  const auto *ir_rhs = op.getInput(1);
  const auto *ir_output = op.getOutput(0);

  // Create the output tensor in the DOM and obtain its identifier.
  auto out = genTensor(ir_output);
  addToPersistentTensors(out);

  // Get the identifiers of the input tensors in the DOM.
  auto lhs = AF::id(tensorName(ir_lhs));
  auto rhs = AF::id(tensorName(ir_rhs));

  genMultiplication(out->name() + "_" + "multiplication", 0, ir_rhs->getShape(), lhs, rhs, out);
}

void AclCppOpGenerator::visit(mir::ops::SubOp &) { throw AclCppException("NYI"); }

void AclCppOpGenerator::visit_fallback(mir::Operation &) { throw AclCppException("NYI"); }

} // namespace nnc
