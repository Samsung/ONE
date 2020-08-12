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

#define OP_REQUIRES(EXP)                                                                         \
  do                                                                                             \
  {                                                                                              \
    if (!(EXP))                                                                                  \
      throw std::runtime_error("OperationValidator failed at line " + std::to_string(__LINE__)); \
  } while (0)

namespace onert
{
namespace compiler
{

OperationValidator::OperationValidator(const ir::Graph &graph)
    : _graph{graph}, _ctx{graph.operands()}, _current_op_seq_layout{ir::Layout::UNKNOWN}
{
}

void OperationValidator::checkUnaryOp(const ir::Operation &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  // Check if I/O types match
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());

  if (_ctx.at(output_index).info().isDynamic())
    return;

  // Check if I/O shapes match
  OP_REQUIRES(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
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

void OperationValidator::visit(const ir::operation::Abs &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::AvgPool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::AvgPool2D::Input::INPUT)};

  OP_REQUIRES(_ctx.at(ifm_index).shape().rank() == 4);
}

void OperationValidator::visit(const ir::operation::BatchMatMul &node)
{
  const auto lhs_index(node.getInputs().at(ir::operation::BatchMatMul::Input::LHS));
  const auto rhs_index(node.getInputs().at(ir::operation::BatchMatMul::Input::RHS));
  const auto out_index{node.getOutputs().at(0)};

  // Constant lhs and rhs is not implemented yet
  OP_REQUIRES(!_ctx.at(lhs_index).isConstant() && !_ctx.at(rhs_index).isConstant());

  if (_ctx.at(out_index).info().isDynamic())
    return;

  OP_REQUIRES(_ctx.at(lhs_index).shape().rank() <= 4);
  OP_REQUIRES(_ctx.at(rhs_index).shape().rank() <= 4);
  OP_REQUIRES(_ctx.at(lhs_index).shape().rank() >= 2);
  OP_REQUIRES(_ctx.at(rhs_index).shape().rank() >= 2);
}

void OperationValidator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::BatchToSpaceND::Input::INPUT)};
  const auto block_size_index{
      node.getInputs().at(ir::operation::BatchToSpaceND::Input::BLOCK_SIZE)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto input_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  const auto output_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);

  // All requirement as per NNAPI specification.
  OP_REQUIRES(_ctx.at(ifm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(ofm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(block_size_index).shape().rank() == 1);

  OP_REQUIRES(_ctx.at(block_size_index).shape().dim(0) == 2);

  OP_REQUIRES(_ctx.at(block_size_index).isConstant());

  OP_REQUIRES(input_shape.C == output_shape.C);
}

void OperationValidator::visit(const ir::operation::Cast &node)
{
  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(0)};

  OP_REQUIRES(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::Comparison &node)
{
  const auto output_index{node.getOutputs().at(0)};
  // This validator does not check shape. So checking isDynamic() is skipped.

  const auto lhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto rhs_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::BOOL8);
}

void OperationValidator::visit(const ir::operation::Softmax &node)
{
  VERBOSE(Softmax) << "Configure SOFTMAX operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(0)};

  OP_REQUIRES(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::InstanceNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::InstanceNorm::Input::INPUT)};
  const auto gamma_index{node.getInputs().at(ir::operation::InstanceNorm::Input::GAMMA)};
  const auto beta_index{node.getInputs().at(ir::operation::InstanceNorm::Input::BETA)};

  OP_REQUIRES(_ctx.at(ifm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(ifm_index).shape() == _ctx.at(ofm_index).shape());
  OP_REQUIRES(_ctx.at(gamma_index).shape().rank() == 1);
  OP_REQUIRES(_ctx.at(beta_index).shape().rank() == 1);
}

void OperationValidator::visit(const ir::operation::Permute &node)
{
  VERBOSE(Permute) << "Configure Permute operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(0)};

  OP_REQUIRES(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::Reduce &node)
{
  VERBOSE(Permute) << "Configure " + node.name() + " operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto input_shape = _ctx.at(input_index).shape();
  const auto output_shape = _ctx.at(output_index).shape();

  OP_REQUIRES(input_shape.rank() <= 4);
  OP_REQUIRES(output_shape.rank() <= input_shape.rank());

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
      OP_REQUIRES(input_shape.dim(0) == output_shape.dim(0) &&
                  input_shape.dim(3) == output_shape.dim(1));
    }
    else if (output_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(input and output) == 1) or (Reducing W and C(input and output) == 1)
      OP_REQUIRES((input_shape.dim(0) == output_shape.dim(0) &&
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
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto &perm{node.param().perm};

  const auto &output_shape = _ctx.at(output_index).shape();
  const auto &input_shape = _ctx.at(input_index).shape();

  OP_REQUIRES(input_shape.rank() == static_cast<int>(perm.size()));
  OP_REQUIRES(input_shape.rank() == output_shape.rank());
}

void OperationValidator::visit(const ir::operation::RNN &node)
{
  // NOTE This validation is for static rnn(non-dynamic shape), but not for dynamic rnn
  // TODO Support dynamic rnn
  const auto output_index{node.getOutputs().at(ir::operation::RNN::Output::OUTPUT)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

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

  OP_REQUIRES(_ctx.at(output_index).shape().rank() == 2 &&
              _ctx.at(hidden_state_out_index).shape().rank() == 2 &&
              _ctx.at(input_index).shape().rank() == 2 &&
              _ctx.at(weights_index).shape().rank() == 2 &&
              _ctx.at(recurrent_weights_index).shape().rank() == 2 &&
              _ctx.at(hidden_state_in_index).shape().rank() == 2);
  OP_REQUIRES(_ctx.at(bias_index).shape().rank() == 1);

  OP_REQUIRES(batch_size == _ctx.at(input_index).shape().dim(0) &&
              batch_size == _ctx.at(hidden_state_in_index).shape().dim(0) &&
              batch_size == _ctx.at(hidden_state_out_index).shape().dim(0));
  OP_REQUIRES(_ctx.at(input_index).shape().dim(1) == _ctx.at(weights_index).shape().dim(1));

  OP_REQUIRES(num_units == _ctx.at(weights_index).shape().dim(0) &&
              num_units == _ctx.at(recurrent_weights_index).shape().dim(0) &&
              num_units == _ctx.at(bias_index).shape().dim(0));
  OP_REQUIRES(num_units == _ctx.at(output_index).shape().dim(1) &&
              num_units == _ctx.at(recurrent_weights_index).shape().dim(1) &&
              num_units == _ctx.at(hidden_state_in_index).shape().dim(1) &&
              num_units == _ctx.at(hidden_state_out_index).shape().dim(1));
}

void OperationValidator::visit(const ir::operation::Round &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  const auto block_size_index{
      node.getInputs().at(ir::operation::SpaceToBatchND::Input::BLOCK_SIZE)};
  const auto paddings_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::PADDINGS)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto input_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  const auto output_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);

  // All requirement as per NNAPI specification.
  OP_REQUIRES(_ctx.at(ifm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(ofm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(block_size_index).shape().rank() == 1);
  OP_REQUIRES(_ctx.at(paddings_index).shape().rank() == 2);

  OP_REQUIRES(_ctx.at(block_size_index).shape().dim(0) == 2);
  OP_REQUIRES(_ctx.at(paddings_index).shape().dim(0) == 2);
  OP_REQUIRES(_ctx.at(paddings_index).shape().dim(1) == 2);

  OP_REQUIRES(_ctx.at(block_size_index).isConstant());
  OP_REQUIRES(_ctx.at(paddings_index).isConstant());

  OP_REQUIRES(input_shape.C == output_shape.C);
}

void OperationValidator::visit(const ir::operation::SpaceToDepth &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToDepth::Input::INPUT)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto input_shape = _ctx.at(ifm_index).shape().asFeature(frontend_layout);
  const auto output_shape = _ctx.at(ofm_index).shape().asFeature(frontend_layout);
  const auto block_size = node.param().block_size;

  // All assertions as per NNAPI specification.
  OP_REQUIRES(_ctx.at(ifm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(ofm_index).shape().rank() == 4);
  OP_REQUIRES((block_size >= 1) && (input_shape.H % block_size == 0) &&
              (input_shape.W % block_size == 0));
  OP_REQUIRES(input_shape.N == output_shape.N);
  OP_REQUIRES(input_shape.C * block_size * block_size == output_shape.C);
}

void OperationValidator::visit(const ir::operation::ElementwiseBinary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::RHS)};

  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(output_index).typeInfo().type());
}

void OperationValidator::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lookups_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::LOOKUPS)};
  const auto values_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::VALUES)};

  const auto &output_obj = _ctx.at(output_index);
  const auto &lookups_obj = _ctx.at(lookups_index);
  const auto &values_obj = _ctx.at(values_index);

  // Verify operand here, not at SimpleEmbeddingLookup::configure() to avoid acl's modifying
  // TensorShape sometimes(Issue: https://github.sec.samsung.net/STAR/nnfw/issues/729)
  {
    OP_REQUIRES(lookups_obj.typeInfo().type() == ir::DataType::INT32);

    if (_ctx.at(output_index).info().isDynamic())
      return;

    const auto &output_shape = output_obj.shape();
    const auto &lookups_shape = lookups_obj.shape();
    const auto &values_shape = values_obj.shape();

    OP_REQUIRES(lookups_shape.rank() == 1);
    OP_REQUIRES(values_shape.rank() >= 2);

    // output should be a n-D tensor with the same rank and shape as the values tensor, except for
    // the first dimension which has the same size as lookups' only dimension.
    OP_REQUIRES(output_shape.rank() == values_shape.rank());
    OP_REQUIRES(output_shape.dim(0) == lookups_shape.dim(0));
    for (int n = 1; n < output_shape.rank(); ++n)
    {
      OP_REQUIRES(output_shape.dim(n) == values_shape.dim(n));
    }
  }
}

void OperationValidator::visit(const ir::operation::Exp &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ExpandDims::Input::AXIS)};

  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(axis_index).typeInfo().type() == ir::DataType::INT32);

  if (_ctx.at(axis_index).info().isDynamic())
    return;
  OP_REQUIRES(_ctx.at(axis_index).shape().rank() <= 1);
}

void OperationValidator::visit(const ir::operation::Floor &node) { checkUnaryOp(node); }

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

  OP_REQUIRES(lookups_obj.typeInfo().type() == ir::DataType::INT32);
  OP_REQUIRES(keys_obj.typeInfo().type() == ir::DataType::INT32);
  OP_REQUIRES(hits_obj.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM);

  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto &output_shape = output_obj.shape();
  const auto &lookups_shape = lookups_obj.shape();
  const auto &keys_shape = keys_obj.shape();
  const auto &values_shape = values_obj.shape();

  OP_REQUIRES(values_shape.rank() == output_shape.rank());
  OP_REQUIRES(lookups_shape.rank() == 1);
  OP_REQUIRES(keys_shape.rank() == 1);
  OP_REQUIRES(values_shape.dim(0) == keys_shape.dim(0));
  OP_REQUIRES(lookups_shape.dim(0) == output_shape.dim(0));
}

void OperationValidator::visit(const ir::operation::TransposeConv &node)
{
  // param check
  OP_REQUIRES((node.param().padding.type == ir::PaddingType::SAME) ||
              (node.param().padding.type == ir::PaddingType::VALID));

  // shape check
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::TransposeConv::Input::INPUT)};
  const auto ker_index{node.getInputs().at(ir::operation::TransposeConv::Input::KERNEL)};

  // Only 4D tensors are supported
  OP_REQUIRES(_ctx.at(ofm_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(ofm_index).shape().rank() == _ctx.at(ifm_index).shape().rank());
  OP_REQUIRES(_ctx.at(ofm_index).shape().rank() == _ctx.at(ker_index).shape().rank());

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

  OP_REQUIRES(ifm_shape.N == ofm_shape.N);
  OP_REQUIRES(ifm_shape.C == ker_shape.C);
  OP_REQUIRES(ker_shape.N == ofm_shape.C);
}

void OperationValidator::visit(const ir::operation::Gather &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  const auto indices_shape = _ctx.at(indices_index).shape();
  const auto ofm_shape = _ctx.at(ofm_index).shape();

  OP_REQUIRES(ifm_shape.rank() <= 4);
  OP_REQUIRES(indices_shape.rank() <= 3);
  OP_REQUIRES(ofm_shape.rank() <= 4);
}

void OperationValidator::visit(const ir::operation::Dequantize &node)
{
  const auto output_index{node.getOutputs().at(0)};

  const auto input_index{node.getInputs().at(ir::operation::Dequantize::Input::INPUT)};

  OP_REQUIRES(_ctx.at(input_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM);
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::FLOAT32);

  if (_ctx.at(output_index).info().isDynamic())
    return;
  OP_REQUIRES(_ctx.at(input_index).shape().rank() <= 4);
  OP_REQUIRES(_ctx.at(input_index).shape() == _ctx.at(output_index).shape());
}

void OperationValidator::visit(const ir::operation::DepthToSpace &node)
{
  // param check
  int32_t block_size = node.param().block_size;

  OP_REQUIRES(block_size > 0);

  // shape check
  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(ir::operation::DepthToSpace::Input::INPUT)};

  const auto frontend_layout = _current_op_seq_layout;
  const auto output_shape = _ctx.at(output_index).shape().asFeature(frontend_layout);
  const auto input_shape = _ctx.at(input_index).shape().asFeature(frontend_layout);

  OP_REQUIRES(_ctx.at(input_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(output_index).shape().rank() == 4);

  {
    OP_REQUIRES(output_shape.N == input_shape.N);
    OP_REQUIRES(output_shape.H == input_shape.H * block_size);
    OP_REQUIRES(output_shape.W == input_shape.W * block_size);
    OP_REQUIRES(input_shape.C % (block_size * block_size) == 0);
    OP_REQUIRES(output_shape.C == input_shape.C / (block_size * block_size));
  }
}

void OperationValidator::visit(const ir::operation::Pack &node)
{
  // param check
  const auto num{node.param().num};
  const auto axis{node.param().axis};
  OP_REQUIRES(num == static_cast<int32_t>(node.getInputs().size()));

  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  // shape check
  const auto &output_shape = _ctx.at(output_index).shape();
  const auto output_rank = static_cast<int32_t>(output_shape.rank());

  const auto input1_index{node.getInputs().at(0)};
  const auto input_shape = _ctx.at(input1_index).shape();

  OP_REQUIRES(axis >= -output_rank && axis < output_rank);
  for (const auto &index : node.getInputs())
  {
    OP_REQUIRES(input_shape == _ctx.at(index).shape());
  }
}

void OperationValidator::visit(const ir::operation::LSTM &node)
{
  // NOTE This validation is for static rnn(non-dynamic shape), but not for dynamic rnn
  // TODO Support dynamic rnn
  const auto output_index{node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto scratch_buffer_index{
      node.getOutputs().at(ir::operation::LSTM::Output::SCRATCH_BUFFER)};
  const auto output_state_out_index{
      node.getOutputs().at(ir::operation::LSTM::Output::OUTPUT_STATE_OUT)};
  const auto cell_state_out_index{
      node.getOutputs().at(ir::operation::LSTM::Output::CELL_STATE_OUT)};

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

  OP_REQUIRES(_ctx.at(scratch_buffer_index).shape().rank() == 2 &&
              _ctx.at(output_state_out_index).shape().rank() == 2 &&
              _ctx.at(cell_state_out_index).shape().rank() == 2 &&
              _ctx.at(output_index).shape().rank() == 2 &&
              _ctx.at(input_index).shape().rank() == 2 &&
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

  OP_REQUIRES(_ctx.at(cell_to_input_weights_index).shape().rank() == 1 &&
              _ctx.at(cell_to_forget_weights_index).shape().rank() == 1 &&
              _ctx.at(cell_to_output_weights_index).shape().rank() == 1 &&
              _ctx.at(input_gate_bias_index).shape().rank() == 1 &&
              _ctx.at(forget_gate_bias_index).shape().rank() == 1 &&
              _ctx.at(cell_bias_index).shape().rank() == 1 &&
              _ctx.at(output_gate_bias_index).shape().rank() == 1 &&
              _ctx.at(projection_bias_index).shape().rank() == 1);

  // CIFG assertion
  OP_REQUIRES((_ctx.at(input_to_input_weights_index).shape().dim(0) == 0 &&
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
  OP_REQUIRES((_ctx.at(cell_to_forget_weights_index).shape().dim(0) == 0 &&
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

  const auto batch_size = _ctx.at(input_index).shape().dim(0);
  OP_REQUIRES(batch_size == _ctx.at(output_state_in_index).shape().dim(0) &&
              batch_size == _ctx.at(cell_state_in_index).shape().dim(0) &&
              batch_size == _ctx.at(scratch_buffer_index).shape().dim(0) &&
              batch_size == _ctx.at(output_state_out_index).shape().dim(0) &&
              batch_size == _ctx.at(cell_state_out_index).shape().dim(0) &&
              batch_size == _ctx.at(output_index).shape().dim(0));

  const auto input_size = _ctx.at(input_index).shape().dim(1);
  OP_REQUIRES(input_size == _ctx.at(input_to_forget_weights_index).shape().dim(1) &&
              input_size == _ctx.at(input_to_cell_weights_index).shape().dim(1) &&
              input_size == _ctx.at(input_to_output_weights_index).shape().dim(1));

  const auto num_units = _ctx.at(cell_state_out_index).shape().dim(1);
  OP_REQUIRES(num_units == _ctx.at(input_to_forget_weights_index).shape().dim(0) &&
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
  OP_REQUIRES(output_size == _ctx.at(recurrent_to_forget_weights_index).shape().dim(1) &&
              output_size == _ctx.at(recurrent_to_cell_weights_index).shape().dim(1) &&
              output_size == _ctx.at(recurrent_to_output_weights_index).shape().dim(1) &&
              output_size == _ctx.at(output_state_in_index).shape().dim(1) &&
              output_size == _ctx.at(output_state_out_index).shape().dim(1));

  if (has_cifg_param)
  {
    OP_REQUIRES(input_size == _ctx.at(input_to_input_weights_index).shape().dim(1));
    OP_REQUIRES(num_units == _ctx.at(input_to_input_weights_index).shape().dim(0) &&
                num_units == _ctx.at(recurrent_to_input_weights_index).shape().dim(0) &&
                (num_units == _ctx.at(cell_to_input_weights_index).shape().dim(0) ||
                 _ctx.at(cell_to_input_weights_index).shape().dim(0) == 0 /* non-peephole */) &&
                num_units == _ctx.at(input_gate_bias_index).shape().dim(0));
    OP_REQUIRES(output_size == _ctx.at(recurrent_to_input_weights_index).shape().dim(1));
    OP_REQUIRES(has_input_to_input_weights && has_recurrent_to_input_weights &&
                has_input_gate_bias);
    if (has_cell_to_input_weights)
    {
      // NOTE The cell_to_input_weights exist only in case of non-CIFG and peephole.
      OP_REQUIRES(has_peephole_param);
    }
    OP_REQUIRES(_ctx.at(scratch_buffer_index).shape().dim(1) == num_units * 4);
  }
  else
  {
    OP_REQUIRES(_ctx.at(scratch_buffer_index).shape().dim(1) == num_units * 3);
  }

  if (has_peephole_param)
  {
    OP_REQUIRES(num_units == _ctx.at(cell_to_forget_weights_index).shape().dim(0) &&
                num_units == _ctx.at(cell_to_output_weights_index).shape().dim(0) &&
                (num_units == _ctx.at(cell_to_input_weights_index).shape().dim(0) ||
                 _ctx.at(cell_to_input_weights_index).shape().dim(0) == 0 /* CIFG */));
  }

  if (has_projection_param)
  {
    OP_REQUIRES(num_units == _ctx.at(projection_weights_index).shape().dim(1));
    OP_REQUIRES(output_size == _ctx.at(projection_weights_index).shape().dim(0));
    if (has_projection_bias)
    {
      OP_REQUIRES(output_size == _ctx.at(projection_bias_index).shape().dim(0));
    }
  }
}

void OperationValidator::visit(const ir::operation::L2Normalization &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  if (_ctx.at(ofm_index).info().isDynamic())
    return;

  const auto ifm_index{node.getInputs().at(ir::operation::L2Normalization::Input::INPUT)};

  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();

  OP_REQUIRES(ifm_shape.rank() == ofm_shape.rank());

  for (auto i = 0; i < ifm_shape.rank(); i++)
  {
    OP_REQUIRES(ifm_shape.dim(i) == ofm_shape.dim(i));
  }
}

void OperationValidator::visit(const ir::operation::Unpack &node)
{
  const auto num{node.param().num};
  OP_REQUIRES(num == static_cast<int32_t>(node.getOutputs().size()));
  const auto axis{node.param().axis};

  const auto output_index{node.getInputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(ir::operation::Unpack::Input::INPUT)};

  const auto &input_shape = _ctx.at(input_index).shape();
  const auto input_rank = static_cast<int32_t>(input_shape.rank());

  OP_REQUIRES(axis >= -input_rank && axis < input_rank);
}

void OperationValidator::visit(const ir::operation::Pad &node)
{
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  OP_REQUIRES(_ctx.at(pad_index).typeInfo().type() == ir::DataType::INT32);

  const auto output_index{node.getInputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};

  const auto &pad_shape = _ctx.at(pad_index).shape();
  const auto input_rank = static_cast<int32_t>(_ctx.at(input_index).shape().rank());

  OP_REQUIRES(pad_shape.rank() == 2);
  OP_REQUIRES(pad_shape.dim(0) == input_rank);
  OP_REQUIRES(pad_shape.dim(1) == 2);
  OP_REQUIRES(_ctx.at(input_index).shape().rank() == _ctx.at(output_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(0)};
  // This validator does not check shape. So checking isDynamic() is skipped.

  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto input_true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto input_false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};
  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(input_true_index);
  UNUSED_RELEASE(input_false_index);

  OP_REQUIRES(_ctx.at(condition_index).typeInfo().type() == ir::DataType::BOOL8);
}

void OperationValidator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto starts_index{node.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto ends_index{node.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto strides_index{node.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};

  UNUSED_RELEASE(starts_index);
  UNUSED_RELEASE(ends_index);
  UNUSED_RELEASE(strides_index);

  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());

  if (_ctx.at(output_index).info().isDynamic())
    return;

  OP_REQUIRES(_ctx.at(input_index).shape().rank() <= 4);
}

void OperationValidator::visit(const ir::operation::Split &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Split::Input::INPUT)};

  if (_ctx.at(input_index).info().isDynamic())
    return;

  const auto num_splits = node.param().num_splits;
  const auto input_rank = _ctx.at(input_index).shape().rank();
  const auto axis = node.param().axis < 0 ? node.param().axis + input_rank : node.param().axis;

  OP_REQUIRES(num_splits > 0 && num_splits <= 0xFFFF);
  OP_REQUIRES(axis >= 0 && axis < input_rank);
  OP_REQUIRES(node.getOutputs().size() == static_cast<uint32_t>(num_splits));

  OP_REQUIRES(_ctx.at(input_index).shape().dim(axis) % num_splits == 0);
}

void OperationValidator::visit(const ir::operation::Cos &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::Sin &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::RSQRT &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::Shape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(0)};
  UNUSED_RELEASE(input_index);
  OP_REQUIRES(_ctx.at(output_index).shape().rank() == 1);
}

void OperationValidator::visit(const ir::operation::ResizeBilinear &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ResizeBilinear::Input::INPUT)};

  if (_ctx.at(output_index).info().isDynamic())
  {
    return;
  }
  OP_REQUIRES(_ctx.at(input_index).shape().rank() == 4);
  OP_REQUIRES(_ctx.at(output_index).shape().rank() == 4);

  auto align_corners = node.param().align_corners;
  auto half_pixel_centers = node.param().half_pixel_centers;

  OP_REQUIRES(!align_corners || !half_pixel_centers);
}

void OperationValidator::visit(const ir::operation::Reverse &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reverse::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Reverse::Input::AXIS)};

  OP_REQUIRES(_ctx.at(axis_index).typeInfo().type() == ir::DataType::INT32);
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(input_index).typeInfo().type());

  if (_ctx.at(output_index).info().isDynamic())
    return;
  OP_REQUIRES(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());
}

void OperationValidator::visit(const ir::operation::If &)
{
  // TODO Add to validate with subgraphs
}

void OperationValidator::visit(const ir::operation::While &node)
{
  // This validator does not check shape. So checking isDynamic() is skipped.

  OP_REQUIRES(node.getInputs().size() == node.getOutputs().size());
  // TODO Add to validate with subgraphs
}

void OperationValidator::visit(const ir::operation::Neg &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::Log &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::LogicalNot &node) { checkUnaryOp(node); }

void OperationValidator::visit(const ir::operation::SquaredDifference &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  // Check for Type equivalence
  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == _ctx.at(lhs_index).typeInfo().type());
  OP_REQUIRES(_ctx.at(lhs_index).typeInfo().type() == _ctx.at(rhs_index).typeInfo().type());

  // Check for dimension constraints
  if (_ctx.at(output_index).info().isDynamic())
    return;

  auto output_shape = _ctx.at(output_index).shape();
  auto lhs_shape = _ctx.at(lhs_index).shape();
  auto rhs_shape = _ctx.at(rhs_index).shape();
  // Check for output rank
  OP_REQUIRES(output_shape.rank() == std::max(lhs_shape.rank(), rhs_shape.rank()));
  auto min_rank = std::min(lhs_shape.rank(), rhs_shape.rank());

  for (int idx = 1; idx <= min_rank; idx++)
  {
    int l_idx = lhs_shape.rank() - idx;
    int r_idx = rhs_shape.rank() - idx;
    int out_idx = output_shape.rank() - idx;

    OP_REQUIRES((l_idx >= 0) && (r_idx >= 0) && (out_idx >= 0));

    auto l_dims = lhs_shape.dim(l_idx);
    auto r_dims = rhs_shape.dim(r_idx);
    auto out_dims = output_shape.dim(out_idx);

    OP_REQUIRES(((l_dims == r_dims) && (out_dims == l_dims)) ||
                ((l_dims == 1) && (out_dims == r_dims)) || ((r_dims == 1) && (out_dims == l_dims)));
  }
  auto &tmp_shape = (lhs_shape.rank() > rhs_shape.rank()) ? lhs_shape : rhs_shape;
  for (int idx = min_rank + 1; idx <= output_shape.rank(); idx++)
  {
    int out_idx = output_shape.rank() - idx;
    int tmp_idx = tmp_shape.rank() - idx;

    OP_REQUIRES((out_idx >= 0) && (tmp_idx >= 0) &&
                (output_shape.dim(out_idx) == tmp_shape.dim(tmp_idx)));
  }
}
void OperationValidator::visit(const ir::operation::Tile &node)
{
  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(0)};
  const auto multiple_index{node.getInputs().at(1)};

  OP_REQUIRES(_ctx.at(multiple_index).shape().rank() == 1);
  OP_REQUIRES(_ctx.at(multiple_index).shape().dim(0) == _ctx.at(input_index).shape().rank());
  OP_REQUIRES(_ctx.at(input_index).shape().rank() == _ctx.at(output_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::Range &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto start_index{node.getInputs().at(ir::operation::Range::Input::START)};
  const auto limit_index{node.getInputs().at(ir::operation::Range::Input::LIMIT)};
  const auto delta_index{node.getInputs().at(ir::operation::Range::Input::DELTA)};

  // Check for dimension constraints
  if (_ctx.at(output_index).info().isDynamic())
    return;

  OP_REQUIRES(_ctx.at(start_index).shape().rank() == 0);
  OP_REQUIRES(_ctx.at(limit_index).shape().rank() == 0);
  OP_REQUIRES(_ctx.at(delta_index).shape().rank() == 0);
}

void OperationValidator::visit(const ir::operation::MatrixBandPart &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::MatrixBandPart::Input::INPUT)};
  const auto num_lower_index{
      node.getInputs().at(ir::operation::MatrixBandPart::Input::NUM_LOWER_DIAG)};
  const auto num_upper_index{
      node.getInputs().at(ir::operation::MatrixBandPart::Input::NUM_UPPER_DIAG)};

  // Check for dimension constraints
  if (_ctx.at(output_index).info().isDynamic())
    return;

  OP_REQUIRES(_ctx.at(input_index).shape().rank() >= 2);     // input must be more than 2 dim matrix
  OP_REQUIRES(_ctx.at(num_upper_index).shape().rank() == 0); // num_lower must be scalar
  OP_REQUIRES(_ctx.at(num_lower_index).shape().rank() == 0); // num_upper must be scalar
}

void OperationValidator::visit(const ir::operation::LogSoftmax &node)
{
  VERBOSE(LogSoftmax) << "Configure LOGSOFTMAX operation" << std::endl;

  const auto output_index{node.getOutputs().at(0)};
  if (_ctx.at(output_index).info().isDynamic())
    return;

  const auto input_index{node.getInputs().at(0)};

  OP_REQUIRES(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
}

void OperationValidator::visit(const ir::operation::Quantize &node)
{
  VERBOSE(Quantize) << "Configure Quantize operation" << std::endl;

  OP_REQUIRES(node.getInputs().size() == 1);
  OP_REQUIRES(node.getOutputs().size() == 1);

  const auto input_index{node.getInputs().at(0)};
  const auto output_index{node.getOutputs().at(0)};

  OP_REQUIRES(_ctx.at(input_index).typeInfo().type() == ir::DataType::FLOAT32);

  if (_ctx.at(output_index).info().isDynamic())
    return;

  OP_REQUIRES(_ctx.at(output_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM);

  OP_REQUIRES(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
}
} // namespace compiler
} // namespace onert
