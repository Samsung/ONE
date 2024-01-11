/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_ACL_COMMON_ACL_KERNEL_GEN_H_
#define __ONERT_BACKEND_ACL_COMMON_ACL_KERNEL_GEN_H_

#include <exec/IFunction.h>
#include <ir/Operands.h>

#include <ir/operation/LSTM.h>
#include <arm_compute/runtime/CL/CLFunctions.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

void enableDimCorrection(IACLTensor *tensor)
{
  size_t input_rank = tensor->getShape().rank();
  const_cast<arm_compute::TensorShape &>(tensor->info()->tensor_shape())
    .set(input_rank - 1, tensor->info()->dimension(input_rank - 1), true);
}

void disableDimCorrection(IACLTensor *tensor)
{
  size_t input_rank = tensor->getShape().rank();
  const_cast<arm_compute::TensorShape &>(tensor->info()->tensor_shape())
    .set(input_rank - 1, tensor->info()->dimension(input_rank - 1), false);
}

template <typename Layer, typename... Args>
std::unique_ptr<arm_compute::IFunction> generateLayer(Args &&...args)
{
  auto l = std::make_unique<Layer>();

  l->configure(std::forward<Args>(args)...);

  return l;
}

template <typename Layer, typename... Args>
std::unique_ptr<arm_compute::IFunction>
generateLayer(std::shared_ptr<arm_compute::IMemoryManager> memory_manager, Args &&...args)
{
  auto l = std::make_unique<Layer>(memory_manager);

  l->configure(std::forward<Args>(args)...);

  return l;
}

template <typename T_FunctionWrapper, typename T_Tensor, typename T_ACLLayer,
          typename T_TensorRegistry>
std::unique_ptr<exec::IFunction> kernelGenLSTM(const ir::operation::LSTM &node,
                                               const ir::Operands &operands,
                                               const std::shared_ptr<T_TensorRegistry> &tensor_reg)
{
  // TODO Support dynamic rnn
  // TODO Fix subtle error in the case of non-CIFG, non-peephole and No Projection.
  const auto scratch_buffer_index{
    node.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};
  const auto output_state_out_index{
    node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};
  const auto cell_state_out_index{
    node.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};
  const auto output_index{node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};

  const auto input_index{node.getInputs().at(ir::operation::LSTM::Input::INPUT)};
  const auto input_to_input_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_INPUT_WEIGHTS)}; // optional
  const auto input_to_forget_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_FORGET_WEIGHTS)};
  const auto input_to_cell_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_CELL_WEIGHTS)};
  const auto input_to_output_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)};
  const auto recurrent_to_input_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)}; // optional
  const auto recurrent_to_forget_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_FORGET_WEIGHTS)};
  const auto recurrent_to_cell_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_CELL_WEIGHTS)};
  const auto recurrent_to_output_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS)};
  const auto cell_to_input_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_INPUT_WEIGHTS)}; // optional
  const auto cell_to_forget_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_FORGET_WEIGHTS)}; // optional
  const auto cell_to_output_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_OUTPUT_WEIGHTS)}; // optional
  const auto input_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::INPUT_GATE_BIAS)};
  const auto forget_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::FORGET_GATE_BIAS)};
  const auto cell_bias_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_BIAS)};
  const auto output_gate_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_GATE_BIAS)};
  const auto projection_weights_index{
    node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_WEIGHTS)}; // optional
  const auto projection_bias_index{
    node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_BIAS)}; // optional
  const auto output_state_in_index{
    node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_STATE_IN)};
  const auto cell_state_in_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_STATE_IN)};
  const auto cell_threshold = node.param().cell_threshold;
  const auto projection_threshold = node.param().projection_threshold;

  bool has_input_to_input_weights = operands.at(input_to_input_weights_index).shape().dim(0) != 0 &&
                                    operands.at(input_to_input_weights_index).shape().dim(1) != 0;
  bool has_recurrent_to_input_weights =
    operands.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
    operands.at(recurrent_to_input_weights_index).shape().dim(1) != 0;
  bool has_cell_to_forget_weights = operands.at(cell_to_forget_weights_index).shape().dim(0) != 0;
  bool has_cell_to_output_weights = operands.at(cell_to_output_weights_index).shape().dim(0) != 0;
  bool has_projection_weights = operands.at(projection_weights_index).shape().dim(0) != 0 &&
                                operands.at(projection_weights_index).shape().dim(1) != 0;
  bool has_projection_bias = operands.at(projection_bias_index).shape().dim(0);

  // NOTE The input_to_input_weights and the recurrent_to_input_weights do not exist in CIFG.
  // true: no CIFG
  // false: CIFG
  // NOTE The cell_to_input_weights does not exist in non-peephole although regular LSTM(non-CIFG).
  bool has_cifg_param = has_input_to_input_weights && has_recurrent_to_input_weights;

  // NOTE The cell_to_forget_weights and the cell_to_output_weights exist in peephole.
  // But the cell_to_input_weights does not exist in regular CIFG although peephole.
  // true: peephole
  // false: no peephole
  bool has_peephole_param = has_cell_to_forget_weights && has_cell_to_output_weights;

  // NOTE Although the projection weights has data the projection bias may not have data.
  bool has_projection_param = has_projection_weights;

  const auto activation = node.param().activation;
  const auto cell_clip = cell_threshold;
  const auto projection_clip = projection_threshold;
  assert(cell_clip >= 0.f && projection_clip >= 0.f);

  auto scratch_buffer_tensor = tensor_reg->getAclTensor(scratch_buffer_index);
  auto output_state_out_tensor = tensor_reg->getAclTensor(output_state_out_index);
  auto cell_state_out_tensor = tensor_reg->getAclTensor(cell_state_out_index);
  auto output_tensor = tensor_reg->getAclTensor(output_index);

  auto input_tensor = tensor_reg->getAclTensor(input_index);

  auto input_to_forget_weights_tensor = tensor_reg->getAclTensor(input_to_forget_weights_index);
  auto input_to_cell_weights_tensor = tensor_reg->getAclTensor(input_to_cell_weights_index);
  auto input_to_output_weights_tensor = tensor_reg->getAclTensor(input_to_output_weights_index);
  auto recurrent_to_forget_weights_tensor =
    tensor_reg->getAclTensor(recurrent_to_forget_weights_index);
  auto recurrent_to_cell_weights_tensor = tensor_reg->getAclTensor(recurrent_to_cell_weights_index);
  auto recurrent_to_output_weights_tensor =
    tensor_reg->getAclTensor(recurrent_to_output_weights_index);

  auto forget_gate_bias_tensor = tensor_reg->getAclTensor(forget_gate_bias_index);
  auto cell_bias_tensor = tensor_reg->getAclTensor(cell_bias_index);
  auto output_gate_bias_tensor = tensor_reg->getAclTensor(output_gate_bias_index);
  auto output_state_in_tensor = tensor_reg->getAclTensor(output_state_in_index);
  auto cell_state_in_tensor = tensor_reg->getAclTensor(cell_state_in_index);

  auto act_info = asActivationLayerInfo(activation);

  ::arm_compute::LSTMParams<T_Tensor> lstm_params{};
  if (has_cifg_param)
  {
    auto input_to_input_weights_tensor =
      tensor_reg->getAclTensor(input_to_input_weights_index); // optional
    auto recurrent_to_input_weights_tensor =
      tensor_reg->getAclTensor(recurrent_to_input_weights_index); // optional
    auto cell_to_input_weights_handle =
      has_peephole_param ? tensor_reg->getAclTensor(cell_to_input_weights_index)->handle()
                         : nullptr; // optional (non-cifg && peephole)
    auto input_gate_bias_tensor = tensor_reg->getAclTensor(input_gate_bias_index); // optional
    lstm_params.set_cifg_params(input_to_input_weights_tensor->handle(),
                                recurrent_to_input_weights_tensor->handle(),
                                cell_to_input_weights_handle, input_gate_bias_tensor->handle());
  }
  if (has_peephole_param)
  {
    auto cell_to_forget_weights_tensor =
      tensor_reg->getAclTensor(cell_to_forget_weights_index); // optional
    auto cell_to_output_weights_tensor =
      tensor_reg->getAclTensor(cell_to_output_weights_index); // optional
    lstm_params.set_peephole_params(cell_to_forget_weights_tensor->handle(),
                                    cell_to_output_weights_tensor->handle());
  }
  if (has_projection_param)
  {
    auto projection_weights_tensor = tensor_reg->getAclTensor(projection_weights_index); // optional
    auto projection_bias_handle = has_projection_bias
                                    ? tensor_reg->getAclTensor(projection_bias_index)->handle()
                                    : nullptr; // optional
    lstm_params.set_projection_params(projection_weights_tensor->handle(), projection_bias_handle);
  }

  auto fn = generateLayer<T_ACLLayer>(
    input_tensor->handle(), input_to_forget_weights_tensor->handle(),
    input_to_cell_weights_tensor->handle(), input_to_output_weights_tensor->handle(),
    recurrent_to_forget_weights_tensor->handle(), recurrent_to_cell_weights_tensor->handle(),
    recurrent_to_output_weights_tensor->handle(), forget_gate_bias_tensor->handle(),
    cell_bias_tensor->handle(), output_gate_bias_tensor->handle(), output_state_in_tensor->handle(),
    cell_state_in_tensor->handle(), scratch_buffer_tensor->handle(),
    output_state_out_tensor->handle(), cell_state_out_tensor->handle(), output_tensor->handle(),
    lstm_params, act_info, cell_clip, projection_clip);

  return std::make_unique<T_FunctionWrapper>(std::move(fn));
}

template <typename T_FunctionWrapper, typename T_Tensor, typename T_ACLLayer,
          typename T_TensorBuilder, typename T_TensorRegistry>
std::unique_ptr<exec::IFunction>
kernelGenFullyConnected(const ir::operation::FullyConnected &node, const ir::Operands &operands,
                        const std::shared_ptr<T_TensorBuilder> &tensor_builder,
                        const std::shared_ptr<T_TensorRegistry> &tensor_reg, ir::Layout layout)
{
  using ir::operation::FullyConnected;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};

  const auto input_rank = operands.at(input_index).shape().rank();

  const auto output_size =
    operands.at(output_index).shape().dim(operands.at(output_index).shape().rank() - 1);
  UNUSED_RELEASE(output_size);
  assert(bias_index.undefined() || operands.at(bias_index).shape().dim(0) == output_size);
  assert(operands.at(weight_index).shape().dim(0) == output_size);
  const auto batch_size =
    operands.at(output_index).shape().dim(operands.at(output_index).shape().rank() - 2);
  const auto input_size =
    operands.at(weight_index).shape().dim(operands.at(weight_index).shape().rank() - 1);

  // Check for reshaping input's shape into rank-2
  bool needs_reshape = false;
  ir::Shape reshape(2);
  if (input_rank == 3 || input_rank == 4)
  {
    const auto &ifm_shape = operands.at(input_index).shape();
    auto feature_size = 1;
    for (int i = 0; i < ifm_shape.rank(); ++i)
    {
      feature_size *= ifm_shape.dim(i);
    }

    UNUSED_RELEASE(feature_size);
    assert(feature_size == batch_size * input_size);

    // for reshaping
    needs_reshape = true;
    reshape.dim(0) = batch_size; /* H */
    reshape.dim(1) = input_size; /* W */
  }

  auto output_tensor = tensor_reg->getAclTensor(output_index);
  const auto input_tensor = tensor_reg->getAclTensor(input_index);
  const auto weight_tensor = tensor_reg->getAclTensor(weight_index);
  const auto bias_tensor = bias_index.undefined() ? nullptr : tensor_reg->getAclTensor(bias_index);
  const auto frontend_layout = layout;
  const auto acl_layout = output_tensor->handle()->info()->data_layout();

  typename T_ACLLayer::KernelType kernel_type = T_ACLLayer::KernelType::GENERAL;
  if (operands.at(weight_index).isConstant())
  {
    kernel_type = T_ACLLayer::KernelType::PREPROCESSED_WEIGHTS;
    assert(operands.at(weight_index).data());
  }

  auto fn = generateLayer<T_ACLLayer>(
    tensor_builder->acl_tensor_manager()->internal_buffer_manager(), input_tensor->handle(),
    weight_tensor->handle(), bias_tensor != nullptr ? bias_tensor->handle() : nullptr,
    output_tensor->handle(), needs_reshape,
    asTensorShape(reshape, frontend_layout, asRuntimeLayout(acl_layout)), kernel_type);

  return std::make_unique<T_FunctionWrapper>(std::move(fn));
}

template <typename T_ACLLayer, typename T_PoolOp, typename T_AclTensorRegistry>
std::unique_ptr<::arm_compute::IFunction>
kernelGenPool2D(const T_PoolOp &node, const ir::Operands &operands,
                const std::shared_ptr<T_AclTensorRegistry> &tensor_reg, ir::Layout layout,
                ::arm_compute::PoolingType pooling_type)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(0)};

  const auto ofm_shape = operands.at(ofm_index).shape().asFeature(layout);
  const auto ifm_shape = operands.at(ifm_index).shape().asFeature(layout);

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto padding =
    ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);

  VERBOSE(Pool2DParam) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(Pool2DParam) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(Pool2DParam) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(Pool2DParam) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(Pool2DParam) << "KER_H: " << kh << std::endl;
  VERBOSE(Pool2DParam) << "KER_W: " << kw << std::endl;
  VERBOSE(Pool2DParam) << "STRIDE_H: " << stride.vertical << std::endl;
  VERBOSE(Pool2DParam) << "STRIDE_W: " << stride.horizontal << std::endl;
  VERBOSE(Pool2DParam) << "PAD(T): " << padding.top << std::endl;
  VERBOSE(Pool2DParam) << "PAD(B): " << padding.bottom << std::endl;
  VERBOSE(Pool2DParam) << "PAD(L): " << padding.left << std::endl;
  VERBOSE(Pool2DParam) << "PAD(R): " << padding.right << std::endl;

  auto ofm_tensor = tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = tensor_reg->getAclTensor(ifm_index);

  ::arm_compute::PoolingLayerInfo info{
    pooling_type, ::arm_compute::Size2D{kw, kh}, ifm_tensor->info()->data_layout(),
    asPadStrideInfo(padding, stride), true /* exclude_padding */};

  auto fn = generateLayer<T_ACLLayer>(ifm_tensor->handle(), ofm_tensor->handle(), info);

  return fn;
}

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_ACL_KERNEL_GEN_H_
