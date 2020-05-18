/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OperationValidator.h"

#include <typeinfo>

#include "ir/Graph.h"
#include "ir/operation/LowerInfo.h"

#include "util/logging.h"
#include "util/Utils.h"

namespace onert
{
namespace compiler
{

OperationValidator::OperationValidator(const ir::Graph &graph)
    : _graph{graph}, _ctx{graph.operands()}, _current_op_seq_layout{ir::Layout::UNKNOWN}
{
}

void OperationValidator::operator()()
{
  // There is no reason for each subgraph to have subgraphs since compiler has subgraphs when
  // creating Compiler
  assert(_graph.subgraphs() == nullptr);

  _current_op_seq_layout = _graph.layout();

  _graph.operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &node) { node.accept(*this); });
}

void OperationValidator::visit(const ir::operation::Abs &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::AvgPool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::AvgPool2D::Input::INPUT)};

  UNUSED_RELEASE(ofm_index);
  UNUSED_RELEASE(ifm_index);

  assert(_ctx.at(ifm_index).shape().rank() == 4);
}

void OperationValidator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::BatchToSpaceND::Input::INPUT)};
  const auto block_size_index{
      node.getInputs().at(ir::operation::BatchToSpaceND::Input::BLOCK_SIZE)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto input_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  const auto output_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);

  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(block_size_index);

  // All assertions as per NNAPI specification.
  assert(_ctx.at(ifm_index).shape().rank() == 4);
  assert(_ctx.at(ofm_index).shape().rank() == 4);
  assert(_ctx.at(block_size_index).shape().rank() == 1);

  assert(_ctx.at(block_size_index).shape().dim(0) == 2);

  assert(_ctx.at(block_size_index).isConstant());

  assert(input_shape.C == output_shape.C);
}

void OperationValidator::visit(const ir::operation::Cast &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(lhs_index);
  UNUSED_RELEASE(rhs_index);

  assert(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  assert(_ctx.at(output_index).typeInfo().type() == ir::DataType::BOOL8);
}

void OperationValidator::visit(const ir::operation::Softmax &node)
{
  VERBOSE(Softmax) << "Configure SOFTMAX operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::InstanceNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::InstanceNorm::Input::INPUT)};
  const auto gamma_index{node.getInputs().at(ir::operation::InstanceNorm::Input::GAMMA)};
  const auto beta_index{node.getInputs().at(ir::operation::InstanceNorm::Input::BETA)};

  UNUSED_RELEASE(ofm_index);
  UNUSED_RELEASE(ifm_index);
  UNUSED_RELEASE(gamma_index);
  UNUSED_RELEASE(beta_index);

  assert(_ctx.at(ifm_index).shape().rank() == 4);
  assert(_ctx.at(ifm_index).shape() == _ctx.at(ofm_index).shape());
  assert(_ctx.at(gamma_index).shape().rank() == 1);
  assert(_ctx.at(beta_index).shape().rank() == 1);
}

void OperationValidator::visit(const ir::operation::Permute &node)
{
  VERBOSE(Permute) << "Configure Permute operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::ReduceSum &node)
{
  VERBOSE(Permute) << "Configure ReduceSum operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceSum::Input::INPUT)};
  const auto &axes = node.param().axes;

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(axes);

  const auto input_shape = _ctx.at(input_index).shape();
  const auto output_shape = _ctx.at(output_index).shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(input_shape);

  assert(input_shape.rank() <= 4);
  assert(output_shape.rank() <= input_shape.rank());

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (input_shape.rank() == 4 && input_shape.rank() != output_shape.rank())
  {
    if (output_shape.rank() == 2)
    {
      // Reducing HW
      assert(input_shape.dim(0) == output_shape.dim(0) &&
             input_shape.dim(3) == output_shape.dim(1));
    }
    else if (output_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(input and output) == 1) or (Reducing W and C(input and output) == 1)
      assert((input_shape.dim(0) == output_shape.dim(0) &&
              input_shape.dim(1) == output_shape.dim(1) &&
              input_shape.dim(2) == output_shape.dim(2)) ||
             (input_shape.dim(0) == output_shape.dim(0) &&
              (input_shape.dim(1) == output_shape.dim(1) ||
               input_shape.dim(2) == output_shape.dim(1)) &&
              input_shape.dim(3) == 1 && output_shape.dim(2) == 1));
    }
  }
}

void OperationValidator::visit(const ir::operation::Transpose &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto &perm{node.param().perm};

  const auto &output_shape = _ctx.at(output_index).shape();
  const auto &input_shape = _ctx.at(input_index).shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(perm);

  assert(input_shape.rank() == static_cast<int>(perm.size()));
  assert(input_shape.rank() == output_shape.rank());
}

void OperationValidator::visit(const ir::operation::ReduceAny &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceAny::Input::INPUT)};
  const auto &axes = node.param().axes;

  auto output_shape = _ctx.at(output_index).shape();
  auto input_shape = _ctx.at(input_index).shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(axes);

  assert(input_shape.rank() <= 4);
  assert(output_shape.rank() <= input_shape.rank());

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (input_shape.rank() == 4 && input_shape.rank() != output_shape.rank())
  {
    if (output_shape.rank() == 2)
    {
      // Reducing HW
      assert(input_shape.dim(0) == output_shape.dim(0) &&
             input_shape.dim(3) == output_shape.dim(1));
    }
    else if (output_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert((input_shape.dim(0) == output_shape.dim(0) &&
              input_shape.dim(1) == output_shape.dim(1) &&
              input_shape.dim(2) == output_shape.dim(2)) ||
             (input_shape.dim(0) == output_shape.dim(0) &&
              (input_shape.dim(1) == output_shape.dim(1) ||
               input_shape.dim(2) == output_shape.dim(1)) &&
              input_shape.dim(3) == 1 && output_shape.dim(2) == 1));
    }
  }
}

void OperationValidator::visit(const ir::operation::ReduceMax &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceMax::Input::INPUT)};
  const auto &axes = node.param().axes;

  auto output_shape = _ctx.at(output_index).shape();
  auto input_shape = _ctx.at(input_index).shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(axes);

  assert(input_shape.rank() <= 4);
  assert(output_shape.rank() <= input_shape.rank());

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (input_shape.rank() == 4 && input_shape.rank() != output_shape.rank())
  {
    if (output_shape.rank() == 2)
    {
      // Reducing HW
      assert(input_shape.dim(0) == output_shape.dim(0) &&
             input_shape.dim(3) == output_shape.dim(1));
    }
    else if (output_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert((input_shape.dim(0) == output_shape.dim(0) &&
              input_shape.dim(1) == output_shape.dim(1) &&
              input_shape.dim(2) == output_shape.dim(2)) ||
             (input_shape.dim(0) == output_shape.dim(0) &&
              (input_shape.dim(1) == output_shape.dim(1) ||
               input_shape.dim(2) == output_shape.dim(1)) &&
              input_shape.dim(3) == 1 && output_shape.dim(2) == 1));
    }
  }
}

void OperationValidator::visit(const ir::operation::RNN &node)
{
  // NOTE This validation is for static rnn(non-dynamic shape), but not for dynamic rnn
  // TODO Support dynamic rnn
  const auto output_index{node.getOutputs().at(ir::operation::RNN::Output::OUTPUT)};
  const auto hidden_state_out_index{
      node.getOutputs().at(ir::operation::RNN::Output::HIDDEN_STATE_OUT)};

  const auto input_index{node.getInputs().at(ir::operation::RNN::Input::INPUT)};
  const auto weights_index{node.getInputs().at(ir::operation::RNN::Input::WEIGHTS)};
  const auto recurrent_weights_index{
      node.getInputs().at(ir::operation::RNN::Input::RECURRENT_WEIGHTS)};
  const auto bias_index{node.getInputs().at(ir::operation::RNN::Input::BIAS)};
  const auto hidden_state_in_index{node.getInputs().at(ir::operation::RNN::Input::HIDDEN_STATE_IN)};

  const auto batch_size = _ctx.at(output_index).shape().dim(0);
  const auto num_units = _ctx.at(output_index).shape().dim(1);

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(hidden_state_out_index);
  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(weights_index);
  UNUSED_RELEASE(recurrent_weights_index);
  UNUSED_RELEASE(bias_index);
  UNUSED_RELEASE(hidden_state_in_index);
  UNUSED_RELEASE(batch_size);
  UNUSED_RELEASE(num_units);

  assert(_ctx.at(output_index).shape().rank() == 2 &&
         _ctx.at(hidden_state_out_index).shape().rank() == 2 &&
         _ctx.at(input_index).shape().rank() == 2 && _ctx.at(weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_weights_index).shape().rank() == 2 &&
         _ctx.at(hidden_state_in_index).shape().rank() == 2);
  assert(_ctx.at(bias_index).shape().rank() == 1);

  assert(batch_size == _ctx.at(input_index).shape().dim(0) &&
         batch_size == _ctx.at(hidden_state_in_index).shape().dim(0) &&
         batch_size == _ctx.at(hidden_state_out_index).shape().dim(0));
  assert(_ctx.at(input_index).shape().dim(1) == _ctx.at(weights_index).shape().dim(1));

  assert(num_units == _ctx.at(weights_index).shape().dim(0) &&
         num_units == _ctx.at(recurrent_weights_index).shape().dim(0) &&
         num_units == _ctx.at(bias_index).shape().dim(0));
  assert(num_units == _ctx.at(output_index).shape().dim(1) &&
         num_units == _ctx.at(recurrent_weights_index).shape().dim(1) &&
         num_units == _ctx.at(hidden_state_in_index).shape().dim(1) &&
         num_units == _ctx.at(hidden_state_out_index).shape().dim(1));
}

void OperationValidator::visit(const ir::operation::Round &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Round::Input::INPUT)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
  assert(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  const auto block_size_index{
      node.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto paddings_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto input_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  const auto output_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);

  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(block_size_index);
  UNUSED_RELEASE(paddings_index);

  // All assertions as per NNAPI specification.
  assert(_ctx.at(ifm_index).shape().rank() == 4);
  assert(_ctx.at(ofm_index).shape().rank() == 4);
  assert(_ctx.at(block_size_index).shape().rank() == 1);
  assert(_ctx.at(paddings_index).shape().rank() == 2);

  assert(_ctx.at(block_size_index).shape().dim(0) == 2);
  assert(_ctx.at(paddings_index).shape().dim(0) == 2);
  assert(_ctx.at(paddings_index).shape().dim(1) == 2);

  assert(_ctx.at(block_size_index).isConstant());
  assert(_ctx.at(paddings_index).isConstant());

  assert(input_shape.C == output_shape.C);
}

void OperationValidator::visit(const ir::operation::SpaceToDepth &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToDepth::Input::INPUT)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto input_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  const auto output_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);
  const auto block_size = node.param().block_size;

  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(block_size);

  // All assertions as per NNAPI specification.
  assert(_ctx.at(ifm_index).shape().rank() == 4);
  assert(_ctx.at(ofm_index).shape().rank() == 4);
  assert((block_size >= 1) && (input_shape.H % block_size == 0) &&
         (input_shape.W % block_size == 0));
  assert(input_shape.N == output_shape.N);
  assert(input_shape.C * block_size * block_size == output_shape.C);
}

void OperationValidator::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lookups_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::LOOKUPS)};
  const auto values_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::VALUES)};

  const auto &output_obj = _ctx.at(output_index);
  const auto &lookups_obj = _ctx.at(lookups_index);
  const auto &values_obj = _ctx.at(values_index);

  UNUSED_RELEASE(output_obj);
  UNUSED_RELEASE(lookups_obj);
  UNUSED_RELEASE(values_obj);

  // Verify operand here, not at SimpleEmbeddingLookup::configure() to avoid acl's modifying
  // TensorShape sometimes(Issue: https://github.sec.samsung.net/STAR/nnfw/issues/729)
  {
    assert(lookups_obj.typeInfo().type() == ir::DataType::INT32);

    const auto &output_shape = output_obj.shape();
    const auto &lookups_shape = lookups_obj.shape();
    const auto &values_shape = values_obj.shape();

    UNUSED_RELEASE(output_shape);
    UNUSED_RELEASE(lookups_shape);
    UNUSED_RELEASE(values_shape);

    assert(lookups_shape.rank() == 1);
    assert(values_shape.rank() >= 2);

    // output should be a n-D tensor with the same rank and shape as the values tensor, except for
    // the first dimension which has the same size as lookups' only dimension.
    assert(output_shape.rank() == values_shape.rank());
    assert(output_shape.dim(0) == lookups_shape.dim(0));
    for (int n = 1; n < output_shape.rank(); ++n)
    {
      assert(output_shape.dim(n) == values_shape.dim(n));
    }
  }
}

void OperationValidator::visit(const ir::operation::Exp &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Exp::Input::INPUT)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
  assert(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(axis_index);

  assert(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
  assert(_ctx.at(axis_index).typeInfo().type() == ir::DataType::INT32);
  assert(_ctx.at(axis_index).shape().rank() <= 1);
}

void OperationValidator::visit(const ir::operation::Fill &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};
  const auto value_index{node.getInputs().at(1)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(value_index);
}

void OperationValidator::visit(const ir::operation::Floor &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Floor::Input::INPUT)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
  assert(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::HashtableLookup &node)
{
  const auto output_index{node.getOutputs().at(ir::operation::HashtableLookup::Output::OUTPUT)};
  const auto hits_index{node.getOutputs().at(ir::operation::HashtableLookup::Output::HITS)};

  const auto lookups_index{node.getInputs().at(ir::operation::HashtableLookup::Input::LOOKUPS)};
  const auto keys_index{node.getInputs().at(ir::operation::HashtableLookup::Input::KEYS)};
  const auto values_index{node.getInputs().at(ir::operation::HashtableLookup::Input::VALUES)};

  const auto &output_obj = _ctx.at(output_index);
  const auto &hits_obj = _ctx.at(hits_index);

  const auto &lookups_obj = _ctx.at(lookups_index);
  const auto &keys_obj = _ctx.at(keys_index);
  const auto &values_obj = _ctx.at(values_index);

  assert(lookups_obj.typeInfo().type() == ir::DataType::INT32);
  assert(keys_obj.typeInfo().type() == ir::DataType::INT32);
  assert(hits_obj.typeInfo().type() == ir::DataType::QUANT8_ASYMM);

  const auto &output_shape = output_obj.shape();
  const auto &hits_shape = hits_obj.shape();

  const auto &lookups_shape = lookups_obj.shape();
  const auto &keys_shape = keys_obj.shape();
  const auto &values_shape = values_obj.shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(hits_shape);
  UNUSED_RELEASE(lookups_shape);
  UNUSED_RELEASE(keys_shape);
  UNUSED_RELEASE(values_shape);

  assert(values_shape.rank() == output_shape.rank());
  assert(lookups_shape.rank() == 1);
  assert(keys_shape.rank() == 1);
  assert(values_shape.dim(0) == keys_shape.dim(0));
  assert(lookups_shape.dim(0) == output_shape.dim(0));
}

void OperationValidator::visit(const ir::operation::TransposeConv &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::TransposeConv::Input::INPUT)};
  const auto ker_index{node.getInputs().at(ir::operation::TransposeConv::Input::KERNEL)};

  // Only 4D tensors are supported
  assert(_ctx.at(ofm_index).shape().rank() == 4);
  assert(_ctx.at(ofm_index).shape().rank() == _ctx.at(ifm_index).shape().rank());
  assert(_ctx.at(ofm_index).shape().rank() == _ctx.at(ker_index).shape().rank());

  const auto frontend_layout = _current_op_seq_layout;
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  // The kernel has only IHWO layout on frontend
  // So ker_shape is treated here below
  // I -> N
  // H -> H
  // W -> W
  // O -> C
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature(ir::Layout::NHWC);

  UNUSED_RELEASE(ofm_shape);
  UNUSED_RELEASE(ifm_shape);
  UNUSED_RELEASE(ker_shape);

  assert((node.param().padding.type == ir::PaddingType::SAME) ||
         (node.param().padding.type == ir::PaddingType::VALID));
  assert(ifm_shape.N == ofm_shape.N);
  assert(ifm_shape.C == ker_shape.C);
  assert(ker_shape.N == ofm_shape.C);
}

void OperationValidator::visit(const ir::operation::Gather &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  const auto ifm_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  const auto axis = node.param().axis;

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  const auto indices_shape = _ctx.at(indices_index).shape();
  const auto ofm_shape = _ctx.at(ofm_index).shape();

  UNUSED_RELEASE(ifm_shape);
  UNUSED_RELEASE(indices_shape);
  UNUSED_RELEASE(ofm_shape);
  UNUSED_RELEASE(axis);

  assert(ifm_shape.rank() <= 4);
  assert(indices_shape.rank() <= 3);
  assert(ofm_shape.rank() <= 4);
}

void OperationValidator::visit(const ir::operation::Dequantize &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Dequantize::Input::INPUT)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(input_index).shape().rank() <= 4);
  assert(_ctx.at(input_index).shape() == _ctx.at(output_index).shape());
  assert(_ctx.at(input_index).typeInfo().type() == ir::DataType::QUANT8_ASYMM);
  assert(_ctx.at(output_index).typeInfo().type() == ir::DataType::FLOAT32);
}

void OperationValidator::visit(const ir::operation::Mean &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Mean::Input::INPUT)};

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  const auto ofm_shape = _ctx.at(ofm_index).shape();

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (ifm_shape.rank() == 4 && ifm_shape.rank() != ofm_shape.rank())
  {
    if (ofm_shape.rank() == 2)
    {
      // Reducing HW
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(3) == ofm_shape.dim(1));
    }
    else if (ofm_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert((ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(1) == ofm_shape.dim(1) &&
              ifm_shape.dim(2) == ofm_shape.dim(2)) ||
             (ifm_shape.dim(0) == ofm_shape.dim(0) &&
              (ifm_shape.dim(1) == ofm_shape.dim(1) || ifm_shape.dim(2) == ofm_shape.dim(1)) &&
              ifm_shape.dim(3) == 1 && ofm_shape.dim(2) == 1));
    }
  }
}

void OperationValidator::visit(const ir::operation::DepthToSpace &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::DepthToSpace::Input::INPUT)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto output_shape = _ctx.at(output_index).shape().asFeature(frontend_layout);
  const auto input_shape = _ctx.at(input_index).shape().asFeature(frontend_layout);

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(input_shape);

  assert(_ctx.at(input_index).shape().rank() == 4);
  assert(_ctx.at(output_index).shape().rank() == 4);

  int32_t block_size = node.param().block_size;

  UNUSED_RELEASE(block_size);

  assert(block_size > 0);

  { // assertions block
    assert(output_shape.N == input_shape.N);
    assert(output_shape.H == input_shape.H * block_size);
    assert(output_shape.W == input_shape.W * block_size);
    assert(input_shape.C % (block_size * block_size) == 0);
    assert(output_shape.C == input_shape.C / (block_size * block_size));
  }
}

void OperationValidator::visit(const ir::operation::Pack &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto num{node.param().num};
  const auto axis{node.param().axis};

  const auto &output_shape = _ctx.at(output_index).shape();
  const auto output_rank = static_cast<int32_t>(output_shape.rank());

  const auto input1_index{node.getInputs().at(0)};
  const auto input_shape = _ctx.at(input1_index).shape();

  UNUSED_RELEASE(num);
  UNUSED_RELEASE(axis);
  UNUSED_RELEASE(output_rank);

  assert(num == static_cast<int32_t>(node.getInputs().size()));
  assert(axis >= -output_rank && axis < output_rank);
  for (const auto &index : node.getInputs())
  {
    UNUSED_RELEASE(index);
    assert(input_shape == _ctx.at(index).shape());
  }
}

void OperationValidator::visit(const ir::operation::ReduceMin &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ReduceMin::Input::INPUT)};
  const auto &axes = node.param().axes;

  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();

  UNUSED_RELEASE(ifm_shape);
  UNUSED_RELEASE(ofm_shape);
  UNUSED_RELEASE(axes);

  assert(ifm_shape.rank() <= 4);
  assert(ofm_shape.rank() <= ifm_shape.rank());

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (ifm_shape.rank() == 4 && ifm_shape.rank() != ofm_shape.rank())
  {
    if (ofm_shape.rank() == 2)
    {
      // Reducing HW
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(3) == ofm_shape.dim(1));
    }
    else if (ofm_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert((ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(1) == ofm_shape.dim(1) &&
              ifm_shape.dim(2) == ofm_shape.dim(2)) ||
             (ifm_shape.dim(0) == ofm_shape.dim(0) &&
              (ifm_shape.dim(1) == ofm_shape.dim(1) || ifm_shape.dim(2) == ofm_shape.dim(1)) &&
              ifm_shape.dim(3) == 1 && ofm_shape.dim(2) == 1));
    }
  }
}

void OperationValidator::visit(const ir::operation::LSTM &node)
{
  // NOTE This validation is for static rnn(non-dynamic shape), but not for dynamic rnn
  // TODO Support dynamic rnn
  const auto scratch_buffer_index{
      node.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};
  const auto output_state_out_index{
      node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};
  const auto cell_state_out_index{
      node.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};
  const auto output_index{node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};

  const auto input_index{node.getInputs().at(ir::operation::LSTM::Input::INPUT)};
  const auto input_to_input_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_INPUT_WEIGHTS)};
  const auto input_to_forget_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_FORGET_WEIGHTS)};
  const auto input_to_cell_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_CELL_WEIGHTS)};
  const auto input_to_output_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)};
  const auto recurrent_to_input_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)};
  const auto recurrent_to_forget_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_FORGET_WEIGHTS)};
  const auto recurrent_to_cell_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_CELL_WEIGHTS)};
  const auto recurrent_to_output_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS)};
  const auto cell_to_input_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_INPUT_WEIGHTS)};
  const auto cell_to_forget_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_FORGET_WEIGHTS)};
  const auto cell_to_output_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::CELL_TO_OUTPUT_WEIGHTS)};
  const auto input_gate_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::INPUT_GATE_BIAS)};
  const auto forget_gate_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::FORGET_GATE_BIAS)};
  const auto cell_bias_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_BIAS)};
  const auto output_gate_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_GATE_BIAS)};
  const auto projection_weights_index{
      node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_WEIGHTS)};
  const auto projection_bias_index{
      node.getInputs().at(ir::operation::LSTM::Input::PROJECTION_BIAS)};
  const auto output_state_in_index{
      node.getInputs().at(ir::operation::LSTM::Input::OUTPUT_STATE_IN)};
  const auto cell_state_in_index{node.getInputs().at(ir::operation::LSTM::Input::CELL_STATE_IN)};

  UNUSED_RELEASE(scratch_buffer_index);
  UNUSED_RELEASE(output_state_out_index);
  UNUSED_RELEASE(cell_state_out_index);
  UNUSED_RELEASE(output_index);

  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(input_to_input_weights_index);
  UNUSED_RELEASE(input_to_forget_weights_index);
  UNUSED_RELEASE(input_to_cell_weights_index);
  UNUSED_RELEASE(input_to_output_weights_index);
  UNUSED_RELEASE(recurrent_to_input_weights_index);
  UNUSED_RELEASE(recurrent_to_forget_weights_index);
  UNUSED_RELEASE(recurrent_to_cell_weights_index);
  UNUSED_RELEASE(recurrent_to_output_weights_index);
  UNUSED_RELEASE(cell_to_input_weights_index);
  UNUSED_RELEASE(cell_to_forget_weights_index);
  UNUSED_RELEASE(cell_to_output_weights_index);
  UNUSED_RELEASE(input_gate_bias_index);
  UNUSED_RELEASE(forget_gate_bias_index);
  UNUSED_RELEASE(cell_bias_index);
  UNUSED_RELEASE(output_gate_bias_index);
  UNUSED_RELEASE(projection_weights_index);
  UNUSED_RELEASE(projection_bias_index);
  UNUSED_RELEASE(output_state_in_index);
  UNUSED_RELEASE(cell_state_in_index);

  assert(_ctx.at(scratch_buffer_index).shape().rank() == 2 &&
         _ctx.at(output_state_out_index).shape().rank() == 2 &&
         _ctx.at(cell_state_out_index).shape().rank() == 2 &&
         _ctx.at(output_index).shape().rank() == 2 && _ctx.at(input_index).shape().rank() == 2 &&
         _ctx.at(input_to_input_weights_index).shape().rank() == 2 &&
         _ctx.at(input_to_forget_weights_index).shape().rank() == 2 &&
         _ctx.at(input_to_cell_weights_index).shape().rank() == 2 &&
         _ctx.at(input_to_output_weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_to_input_weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_to_forget_weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_to_cell_weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_to_output_weights_index).shape().rank() == 2 &&
         _ctx.at(projection_weights_index).shape().rank() == 2 &&
         _ctx.at(output_state_in_index).shape().rank() == 2 &&
         _ctx.at(cell_state_in_index).shape().rank() == 2);

  assert(_ctx.at(cell_to_input_weights_index).shape().rank() == 1 &&
         _ctx.at(cell_to_forget_weights_index).shape().rank() == 1 &&
         _ctx.at(cell_to_output_weights_index).shape().rank() == 1 &&
         _ctx.at(input_gate_bias_index).shape().rank() == 1 &&
         _ctx.at(forget_gate_bias_index).shape().rank() == 1 &&
         _ctx.at(cell_bias_index).shape().rank() == 1 &&
         _ctx.at(output_gate_bias_index).shape().rank() == 1 &&
         _ctx.at(projection_bias_index).shape().rank() == 1);

  // CIFG assertion
  assert((_ctx.at(input_to_input_weights_index).shape().dim(0) == 0 &&
          _ctx.at(input_to_input_weights_index).shape().dim(1) == 0 &&
          _ctx.at(recurrent_to_input_weights_index).shape().dim(0) == 0 &&
          _ctx.at(recurrent_to_input_weights_index).shape().dim(1) == 0 &&
          _ctx.at(input_gate_bias_index).shape().dim(0) == 0 &&
          _ctx.at(cell_to_input_weights_index).shape().dim(0) == 0) ||
         (_ctx.at(input_to_input_weights_index).shape().dim(0) != 0 &&
          _ctx.at(input_to_input_weights_index).shape().dim(1) != 0 &&
          _ctx.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
          _ctx.at(recurrent_to_input_weights_index).shape().dim(1) != 0 &&
          _ctx.at(input_gate_bias_index).shape().dim(0) != 0));

  // Peephole assertion
  assert((_ctx.at(cell_to_forget_weights_index).shape().dim(0) == 0 &&
          _ctx.at(cell_to_output_weights_index).shape().dim(0) == 0) ||
         (_ctx.at(cell_to_forget_weights_index).shape().dim(0) != 0 &&
          _ctx.at(cell_to_output_weights_index).shape().dim(0) != 0));

  bool has_input_to_input_weights = _ctx.at(input_to_input_weights_index).shape().dim(0) != 0 &&
                                    _ctx.at(input_to_input_weights_index).shape().dim(1) != 0;
  bool has_recurrent_to_input_weights =
      _ctx.at(recurrent_to_input_weights_index).shape().dim(0) != 0 &&
      _ctx.at(recurrent_to_input_weights_index).shape().dim(1) != 0;
  bool has_input_gate_bias = _ctx.at(input_gate_bias_index).shape().dim(0) != 0;
  bool has_cell_to_input_weights = _ctx.at(cell_to_input_weights_index).shape().dim(0) != 0;
  bool has_cell_to_forget_weights = _ctx.at(cell_to_forget_weights_index).shape().dim(0) != 0;
  bool has_cell_to_output_weights = _ctx.at(cell_to_output_weights_index).shape().dim(0) != 0;
  bool has_projection_weights = _ctx.at(projection_weights_index).shape().dim(0) != 0 &&
                                _ctx.at(projection_weights_index).shape().dim(1) != 0;
  bool has_projection_bias = _ctx.at(projection_bias_index).shape().dim(0);

  // NOTE The cell_to_input_weights do not exist in non-peephole although regular LSTM(non-CIFG).
  // true: no CIFG
  // false: CIFG
  bool has_cifg_param = has_input_to_input_weights && has_recurrent_to_input_weights;

  // NOTE The cell_to_input_weights do not exist in regular CIFG although peephole.
  // true: peephole
  // false: no peephole
  bool has_peephole_param = has_cell_to_forget_weights && has_cell_to_output_weights;

  // NOTE The projection weights may have data but the projection bias may not.
  bool has_projection_param = has_projection_weights;

  UNUSED_RELEASE(has_input_to_input_weights);
  UNUSED_RELEASE(has_recurrent_to_input_weights);
  UNUSED_RELEASE(has_input_gate_bias);
  UNUSED_RELEASE(has_cell_to_input_weights);
  UNUSED_RELEASE(has_cell_to_forget_weights);
  UNUSED_RELEASE(has_cell_to_output_weights);
  UNUSED_RELEASE(has_projection_weights);
  UNUSED_RELEASE(has_projection_bias);
  UNUSED_RELEASE(has_cifg_param);
  UNUSED_RELEASE(has_peephole_param);
  UNUSED_RELEASE(has_projection_param);

  const auto batch_size = _ctx.at(input_index).shape().dim(0);
  UNUSED_RELEASE(batch_size);
  assert(batch_size == _ctx.at(output_state_in_index).shape().dim(0) &&
         batch_size == _ctx.at(cell_state_in_index).shape().dim(0) &&
         batch_size == _ctx.at(scratch_buffer_index).shape().dim(0) &&
         batch_size == _ctx.at(output_state_out_index).shape().dim(0) &&
         batch_size == _ctx.at(cell_state_out_index).shape().dim(0) &&
         batch_size == _ctx.at(output_index).shape().dim(0));

  const auto input_size = _ctx.at(input_index).shape().dim(1);
  UNUSED_RELEASE(input_size);
  assert(input_size == _ctx.at(input_to_forget_weights_index).shape().dim(1) &&
         input_size == _ctx.at(input_to_cell_weights_index).shape().dim(1) &&
         input_size == _ctx.at(input_to_output_weights_index).shape().dim(1));

  const auto num_units = _ctx.at(cell_state_out_index).shape().dim(1);
  UNUSED_RELEASE(num_units);
  assert(num_units == _ctx.at(input_to_forget_weights_index).shape().dim(0) &&
         num_units == _ctx.at(input_to_cell_weights_index).shape().dim(0) &&
         num_units == _ctx.at(input_to_output_weights_index).shape().dim(0) &&
         num_units == _ctx.at(recurrent_to_forget_weights_index).shape().dim(0) &&
         num_units == _ctx.at(recurrent_to_cell_weights_index).shape().dim(0) &&
         num_units == _ctx.at(recurrent_to_output_weights_index).shape().dim(0) &&
         num_units == _ctx.at(forget_gate_bias_index).shape().dim(0) &&
         num_units == _ctx.at(cell_bias_index).shape().dim(0) &&
         num_units == _ctx.at(output_gate_bias_index).shape().dim(0) &&
         num_units == _ctx.at(cell_state_in_index).shape().dim(1) &&
         (((num_units * 3) == _ctx.at(scratch_buffer_index).shape().dim(1)) ||
          ((num_units * 4) == _ctx.at(scratch_buffer_index).shape().dim(1))));

  const auto output_size = _ctx.at(output_index).shape().dim(1);
  UNUSED_RELEASE(output_size);
  assert(output_size == _ctx.at(recurrent_to_forget_weights_index).shape().dim(1) &&
         output_size == _ctx.at(recurrent_to_cell_weights_index).shape().dim(1) &&
         output_size == _ctx.at(recurrent_to_output_weights_index).shape().dim(1) &&
         output_size == _ctx.at(output_state_in_index).shape().dim(1) &&
         output_size == _ctx.at(output_state_out_index).shape().dim(1));

  if (has_cifg_param)
  {
    assert(input_size == _ctx.at(input_to_input_weights_index).shape().dim(1));
    assert(num_units == _ctx.at(input_to_input_weights_index).shape().dim(0) &&
           num_units == _ctx.at(recurrent_to_input_weights_index).shape().dim(0) &&
           (num_units == _ctx.at(cell_to_input_weights_index).shape().dim(0) ||
            _ctx.at(cell_to_input_weights_index).shape().dim(0) == 0 /* non-peephole */) &&
           num_units == _ctx.at(input_gate_bias_index).shape().dim(0));
    assert(output_size == _ctx.at(recurrent_to_input_weights_index).shape().dim(1));
    assert(has_input_to_input_weights && has_recurrent_to_input_weights && has_input_gate_bias);
    if (has_cell_to_input_weights)
    {
      // NOTE The cell_to_input_weights exist only in case of non-CIFG and peephole.
      assert(has_peephole_param);
    }
    assert(_ctx.at(scratch_buffer_index).shape().dim(1) == num_units * 4);
  }
  else
  {
    assert(_ctx.at(scratch_buffer_index).shape().dim(1) == num_units * 3);
  }

  if (has_peephole_param)
  {
    assert(num_units == _ctx.at(cell_to_forget_weights_index).shape().dim(0) &&
           num_units == _ctx.at(cell_to_output_weights_index).shape().dim(0) &&
           (num_units == _ctx.at(cell_to_input_weights_index).shape().dim(0) ||
            _ctx.at(cell_to_input_weights_index).shape().dim(0) == 0 /* CIFG */));
  }

  if (has_projection_param)
  {
    assert(num_units == _ctx.at(projection_weights_index).shape().dim(1));
    assert(output_size == _ctx.at(projection_weights_index).shape().dim(0));
    if (has_projection_bias)
    {
      assert(output_size == _ctx.at(projection_bias_index).shape().dim(0));
    }
  }
}

void OperationValidator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Unpack::Input::INPUT)};
  const auto num{node.param().num};
  const auto axis{node.param().axis};

  const auto &input_shape = _ctx.at(input_index).shape();
  const auto input_rank = static_cast<int32_t>(input_shape.rank());

  UNUSED_RELEASE(num);
  UNUSED_RELEASE(axis);
  UNUSED_RELEASE(input_rank);

  assert(num == static_cast<int32_t>(node.getOutputs().size()));
  assert(axis >= -input_rank && axis < input_rank);
}

void OperationValidator::visit(const ir::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto output_index{node.getOutputs().at(0)};

  const auto &pad_shape = _ctx.at(pad_index).shape();
  const auto input_rank = static_cast<int32_t>(_ctx.at(input_index).shape().rank());

  UNUSED_RELEASE(pad_shape);
  UNUSED_RELEASE(input_rank);
  UNUSED_RELEASE(output_index);

  assert(pad_shape.rank() == 2);
  assert(pad_shape.dim(0) == input_rank);
  assert(pad_shape.dim(1) == 2);
  assert(_ctx.at(pad_index).typeInfo().type() == ir::DataType::INT32);
  assert(_ctx.at(input_index).shape().rank() == _ctx.at(output_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::Min &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Min::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Min::Input::RHS)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(lhs_index);
  UNUSED_RELEASE(rhs_index);

  assert(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  assert(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(output_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::Max &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Max::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Max::Input::RHS)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(lhs_index);
  UNUSED_RELEASE(rhs_index);

  assert(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  assert(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(output_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::Select &node)
{
  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto input_true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto input_false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};
  const auto output_index{node.getOutputs().at(0)};

  UNUSED_RELEASE(condition_index);
  UNUSED_RELEASE(input_true_index);
  UNUSED_RELEASE(input_false_index);
  UNUSED_RELEASE(output_index);

  assert(_ctx.at(condition_index).typeInfo().type() == ir::DataType::BOOL8);
}

void OperationValidator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto starts_index{node.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto ends_index{node.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto strides_index{node.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(starts_index);
  UNUSED_RELEASE(ends_index);
  UNUSED_RELEASE(strides_index);

  assert(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
  assert(_ctx.at(input_index).shape().rank() <= 4);
}

void OperationValidator::visit(const ir::operation::Split &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto &num_splits = node.param().num_splits;
  const auto &input_rank = node.param().rank;
  const auto &axis = node.param().axis < 0 ? node.param().axis + input_rank : node.param().axis;

  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(num_splits);
  UNUSED_RELEASE(input_rank);
  UNUSED_RELEASE(axis);

  assert(num_splits > 0 && num_splits <= 0xFFFF);
  assert(axis >= 0 && axis < input_rank);
  assert(_ctx.at(input_index).shape().dim(axis) % num_splits == 0);
  assert(node.getOutputs().size() == static_cast<uint32_t>(num_splits));
}

void OperationValidator::visit(const ir::operation::Cos &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::Sin &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::RSQRT &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::Shape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape().rank() == 1);
}

void OperationValidator::visit(const ir::operation::ReduceProd &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ReduceProd::Input::INPUT)};
  const auto &axes = node.param().axes;

  auto output_shape = _ctx.at(output_index).shape();
  auto input_shape = _ctx.at(input_index).shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(input_shape);
  UNUSED_RELEASE(axes);

  assert(input_shape.rank() <= 4);
  assert(output_shape.rank() <= input_shape.rank());

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (input_shape.rank() == 4 && input_shape.rank() != output_shape.rank())
  {
    if (output_shape.rank() == 2)
    {
      // Reducing HW
      assert(input_shape.dim(0) == output_shape.dim(0) &&
             input_shape.dim(3) == output_shape.dim(1));
    }
    else if (output_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert((input_shape.dim(0) == output_shape.dim(0) &&
              input_shape.dim(1) == output_shape.dim(1) &&
              input_shape.dim(2) == output_shape.dim(2)) ||
             (input_shape.dim(0) == output_shape.dim(0) &&
              (input_shape.dim(1) == output_shape.dim(1) ||
               input_shape.dim(2) == output_shape.dim(1)) &&
              input_shape.dim(3) == 1 && output_shape.dim(2) == 1));
    }
  }
}

void OperationValidator::visit(const ir::operation::Reverse &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Reverse::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Reverse::Input::AXIS)};
  const auto output_index{node.getOutputs().at(0)};

  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(axis_index);
  UNUSED_RELEASE(output_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
  assert(_ctx.at(axis_index).typeInfo().type() == ir::DataType::INT32);
  assert(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::While &node)
{
  assert(node.getInputs().size() == node.getOutputs().size());
  UNUSED_RELEASE(node);
  // TODO Add to validate with subgraphs
}

void OperationValidator::visit(const ir::operation::Neg &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::Log &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::LogicalNot &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::Tile &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};
  const auto multiple_index{node.getInputs().at(1)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_index);
  UNUSED_RELEASE(multiple_index);

  assert(_ctx.at(multiple_index).shape().rank() == 1);
  assert(_ctx.at(input_index).shape().rank() == _ctx.at(output_index).shape().rank());
}

} // namespace compiler
} // namespace onert
