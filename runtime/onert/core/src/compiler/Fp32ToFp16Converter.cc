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

#if 0 // This file is temporarily unused

#include "Fp32ToFp16Converter.h"
#include "ir/operation/ConvertFp32ToFp16.h"
#include "ir/operation/ConvertFp16ToFp32.h"
#include "util/logging.h"

#include <Half.h>

using float16 = Half;

namespace
{

const std::string kAclClBackendConfigId = "acl_cl";

void copyDataFromFp32ToFp16(const float *from, float16 *into, size_t num_elements)
{
  for (size_t i = 0; i < num_elements; ++i)
  {
    into[i] = static_cast<float16>(from[i]);
  }
}

} // namespace

namespace onert
{

namespace compiler
{

Fp32ToFp16Converter::Fp32ToFp16Converter(compiler::LoweredGraph &lowered_graph)
  : _lowered_graph{lowered_graph}
{
  VERBOSE(Fp32ToFp16Converter) << "Fp16 Enable on" << std::endl;
}

// For example, two OpSequences are there and each OpSequence has an Operation
//
//   OP#0      // model input
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#1
//    |
// [OPERATION] // OpSeq#1
//    |
//   OP#2      // model output
//
//
// AFTER `appendOpSequences()`,
// note that model_input and model_output are not changed.
//
//   OP#0
//    |
// [FP32TO16]  // OpSeq#2
//    |
//   OP#3
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#4
//    |
// [FP16TO32]  // OpSeq#3
//    |
//   OP#1
//    |
// [FP32TO16]  // OpSeq#4
//    |
//   OP#5
//    |
// [OPERATION] // OpSeq#1
//    |
//   OP#6
//    |
// [FP16TO32]  // OpSeq#5
//    |
//   OP#2
//
//
// AFTER `optimize()`,
//
//   OP#0
//    |
// [FP32TO16]  // OpSeq#2
//    |
//   OP#3
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#4
//    |
// [OPERATION] // OpSeq#1
//    |
//   OP#6
//    |
// [FP16TO32]  // OpSeq#5
//    |
//   OP#2
//
//
// AFTER `convertOperands()`,
//
//   OP#0      // model_input, not fp16
//    |
// [FP32TO16]  // OpSeq#2
//    |
//   OP#3      // fp16
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#4      // fp16
//    |
// [OPERATION] // OpSeq#1
//    |
//   OP#6      // fp16
//    |
// [FP16TO32]  // OpSeq#5
//    |
//   OP#2      // model_output, notfp16
//
//
// AFTER `convertDatas()`,
//
//   OP#0      // model_input, not fp16
//    |
// [FP32TO16]  // OpSeq#2
//    |
//   OP#3      // fp16
//    |
// [OPERATION] // OpSeq#0, constants are fp16
//    |
//   OP#4      // fp16
//    |
// [OPERATION] // OpSeq#1, constants are fp16
//    |
//   OP#6      // fp16
//    |
// [FP16TO32]  // OpSeq#5
//    |
//   OP#2      // model_output, notfp16
//
void Fp32ToFp16Converter::run()
{
  // Append new OpSequence which includes ConvertFp32ToFp16
  //   and append new OpSequence which includes ConvertFp16ToFp32
  appendOpSequences();

  // Remove unnecessary converting operations
  optimize();

  // Convert operands' data types from fp32 to fp16
  convertOperands();

  // Convert Datas
  convertDatas();

  // Print the result
  printOpSequences("FINAL OpSequences");
}

void Fp32ToFp16Converter::appendOpSequences()
{
  _lowered_graph.op_seqs().iterate(
    [&](const ir::OpSequenceIndex &op_seq_ind, ir::OpSequence &op_seq) {
      const auto &lower_info = _lowered_graph.getLowerInfo(op_seq_ind);
      assert(lower_info != nullptr);

      // For now, the only acl_cl supports fully fp16 type
      // TODO Support fp16 on acl_neon. Current acl_neon supports the only reshape and concat
      // operations.
      //      To do this, we could check the support by `operation by operation`. After that, we
      //      would partition an op_seq if it contains unsupported operations.
      if (lower_info->backend()->config()->id() != kAclClBackendConfigId)
        return;

      // OpSeq's input set should be included in the first operation's input set or
      // OpSeq's output set should be included in the last operation's output set
      assert(checkOperandsOfOpSequence(op_seq));

      // Append converting OpSequence for fp16 but all operands' types are not fp16 still.
      appendNewOpSeqForConvertFp32ToFp16(op_seq_ind, op_seq);
      appendNewOpSeqForConvertFp16ToFp32(op_seq_ind, op_seq);
    });
}

//
// BEFORE
//
//   OP#0      // model input
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#1      // model output
//
//
// AFTER
//
//   OP#0      // model input
//    |
// [FP32TO16]  // OpSeq#1
//    |
//   OP#2
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#1      // model output
//
void Fp32ToFp16Converter::appendNewOpSeqForConvertFp32ToFp16(const ir::OpSequenceIndex &op_seq_ind,
                                                             ir::OpSequence &op_seq)
{
  // OpSeq's input set is included in the first operation's input set
  const ir::OperandIndexSequence op_seq_inputs = op_seq.getInputs(); // copied

  // NOTE Please do not change sequence of op_seq_inputs. It can change the sequence of inputs of
  // Subgraph
  for (const auto &op_seq_input_ind :
       op_seq_inputs | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    if (checkOperandType(op_seq_input_ind) == false)
      continue;

    // new operand w/ datatype fp32
    const auto new_op_ind = newCopiedOperand(op_seq_input_ind);

    // set new lower_info for operand
    setNewOperandLowerInfo(op_seq_ind, new_op_ind);

    // manipulate input of operation and op_seq
    // - replace the first operation's input to new operand
    //   with old operand's removeUse and new operand's appendUse()
    manipulateInput(op_seq_ind, op_seq_input_ind, new_op_ind);

    // new op
    const auto new_node_ind = newOperationConvertFp32ToFp16(op_seq_input_ind, new_op_ind);

    // new op_seq
    const auto new_op_seq_ind = newOpSequence(op_seq_ind, new_node_ind);

    // set new lower_info for op_seq
    setNewOperationLowerInfo(op_seq_ind, new_op_seq_ind);

    _list_fp32_to_fp16.insert(new_op_seq_ind);

    VERBOSE(Fp32ToFp16Converter) << "NEW   |Fp32To16]"
                                 << ir::getStrFromOpSeq(_lowered_graph.op_seqs().at(new_op_seq_ind),
                                                        _lowered_graph.graph().operations())
                                 << std::endl;
  }
}

//
// BEFORE
//
//   OP#0      // model input
//    |
// [FP32TO16]  // OpSeq#1
//    |
//   OP#2
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#1      // model output
//
//
// AFTER
//
//   OP#0      // model input
//    |
// [FP32TO16]  // OpSeq#1
//    |
//   OP#2
//    |
// [OPERATION] // OpSeq#0
//    |
//   OP#3
//    |
// [FP16TO32]  // OpSeq#2
//    |
//   OP#1      // model output
//
void Fp32ToFp16Converter::appendNewOpSeqForConvertFp16ToFp32(const ir::OpSequenceIndex &op_seq_ind,
                                                             ir::OpSequence &op_seq)
{
  // OpSeq's output set is included in the last operation's output set
  const ir::OperandIndexSequence op_seq_outputs = op_seq.getOutputs(); // copied

  // NOTE Please do not change sequence of op_seq_outputs. It can change the sequence of outputs of
  // Subgraph
  for (const auto &op_seq_output_ind :
       op_seq_outputs | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
  {
    if (checkOperandType(op_seq_output_ind) == false)
      continue;

    // new operand w/ datatype fp32
    const auto new_op_ind = newCopiedOperand(op_seq_output_ind);

    // set new lower_info for operand
    setNewOperandLowerInfo(op_seq_ind, new_op_ind);

    // manipulate output of operation and op_seq
    // - replace output of the last operation's output to new operand
    //    with old operand's unsetDef and new operand's appendDef()
    manipulateOutput(op_seq_ind, op_seq_output_ind, new_op_ind);

    // new op
    auto new_node_ind = newOperationConvertFp16ToFp32(op_seq_output_ind, new_op_ind);

    // new op_seq
    auto new_op_seq_ind = newOpSequence(op_seq_ind, new_node_ind);

    // set new lower_info for op_seq
    setNewOperationLowerInfo(op_seq_ind, new_op_seq_ind);

    _list_fp16_to_fp32.insert(new_op_seq_ind);

    VERBOSE(Fp32ToFp16Converter) << "NEW   |Fp16To32]"
                                 << ir::getStrFromOpSeq(_lowered_graph.op_seqs().at(new_op_seq_ind),
                                                        _lowered_graph.graph().operations())
                                 << std::endl;
  }
}

void Fp32ToFp16Converter::optimize()
{
  printOpSequences("BEFORE opt");

  removeContiguousConvertOpSequences();

  printOpSequences("AFTER removeContiguousConverts");

  // TODO Handle Split from the beginning of the model. ex) MODELS/inception_module
  //
  // BEFORE)
  //
  //   OP#0---------------------.         // model_input
  //    |                       |
  // [FP32TO16]  // OpSeq#0   [FP32TO16]  // OpSeq#1
  //    |                       |
  //   OP#1                    OP#2
  //    |                       |
  // [OPERATION] // OpSeq#2   [OPERATION] // OpSeq#3
  //
  //
  // AFTER)
  //
  //   OP#0      // model_input
  //    |
  // [FP32TO16]  // OpSeq#4
  //    |
  //   OP#3---------------------------.
  //    |                             |
  // [OPERATION] // OpSeq#2   [OPERATION] // OpSeq#3
}

void Fp32ToFp16Converter::convertOperands()
{
  _lowered_graph.op_seqs().iterate(
    [&](const ir::OpSequenceIndex &op_seq_ind, ir::OpSequence &op_seq) {
      const auto &lower_info = _lowered_graph.getLowerInfo(op_seq_ind);
      assert(lower_info != nullptr);
      // For now, the only acl_cl supports fully fp16
      if (lower_info->backend()->config()->id() != kAclClBackendConfigId)
        return;

      // Convert input,output operands' type to fp16
      convertOperandsOfOpSequence(op_seq);
    });
}

void Fp32ToFp16Converter::convertOperandsOfOpSequence(ir::OpSequence &op_seq)
{
  auto &operands = _lowered_graph.graph().operands();
  const auto &operations = _lowered_graph.graph().operations();
  const auto &op_seq_inputs = _lowered_graph.graph().getInputs();
  const auto &op_seq_outputs = _lowered_graph.graph().getOutputs();

  for (const auto &op_idx : op_seq)
  {
    const auto &node = operations.at(op_idx);
    for (const auto &ind : node.getInputs() | ir::Remove::UNDEFINED)
    {
      if (node.opcode() == ir::OpCode::ConvertFp32ToFp16 || op_seq_inputs.contains(ind))
        continue;

      auto &obj = operands.at(ind);
      if (obj.isConstant() || obj.typeInfo().type() != ir::DataType::FLOAT32)
        continue;

      obj.type(ir::DataType::FLOAT16);

      VERBOSE(Fp32ToFp16Converter) << "Input Operand " << ind << ": fp16" << std::endl;
    }

    for (const auto &ind : node.getOutputs())
    {
      if (node.opcode() == ir::OpCode::ConvertFp16ToFp32 || op_seq_outputs.contains(ind))
        continue;

      auto &obj = operands.at(ind);
      if (obj.isConstant() || obj.typeInfo().type() != ir::DataType::FLOAT32)
        continue;

      obj.type(ir::DataType::FLOAT16);

      VERBOSE(Fp32ToFp16Converter) << "Output Operand " << ind << ": fp16" << std::endl;
    }
  }
}

void Fp32ToFp16Converter::convertDatas()
{
  _lowered_graph.graph().operands().iterate([&](const ir::OperandIndex &ind, ir::Operand &obj) {
    const auto type = obj.typeInfo().type();
    if (type == ir::DataType::FLOAT32 && obj.isConstant())
    {
      auto data = obj.data();
      assert(data != nullptr);

      size_t num_elements = obj.operandSize() / ir::sizeOfDataType(type);
      size_t new_ptr_size = num_elements * sizeof(float16);
      auto new_ptr = std::make_unique<uint8_t[]>(new_ptr_size);
      copyDataFromFp32ToFp16(reinterpret_cast<const float *>(data->base()),
                             reinterpret_cast<float16 *>(new_ptr.get()), num_elements);
      obj.releaseData();

      auto new_data = std::make_unique<ir::CachedData>(new_ptr.get(), new_ptr_size);

      obj.data(std::move(new_data));
      obj.type(ir::DataType::FLOAT16);
      VERBOSE(Fp32ToFp16Converter) << "Constant Operand " << ind << ": fp16" << std::endl;
    }
  });
}

void Fp32ToFp16Converter::printOpSequences(const std::string &pre_msg, const std::string &post_msg)
{
  if (pre_msg.empty() == false)
  {
    VERBOSE(Fp32ToFp16Converter) << pre_msg << std::endl;
  }

  _lowered_graph.op_seqs().iterate([&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
    VERBOSE(Fp32ToFp16Converter) << ir::getStrFromOpSeq(op_seq, _lowered_graph.graph().operations())
                                 << std::endl;
  });

  if (post_msg.empty() == false)
  {
    VERBOSE(Fp32ToFp16Converter) << post_msg << std::endl;
  }
}

bool Fp32ToFp16Converter::checkOperandType(const ir::OperandIndex &op_ind) const
{
  const auto &operands = _lowered_graph.graph().operands();
  const auto &obj = operands.at(op_ind);
  return (obj.isConstant() == false && obj.typeInfo().type() == ir::DataType::FLOAT32);
}

bool Fp32ToFp16Converter::checkOperandsOfOpSequence(const ir::OpSequence &op_seq) const
{
  const auto &operations = _lowered_graph.graph().operations();

  // the first node's input
  const auto &first_node_ind = op_seq.operations().at(0);
  const auto &first_node = operations.at(first_node_ind);
  const auto &first_node_inputs = first_node.getInputs();
  for (const auto &op_seq_input_ind : op_seq.getInputs() | ir::Remove::UNDEFINED)
  {
    if (first_node_inputs.contains(op_seq_input_ind) == false)
      return false;
  }

  // the last node's output
  size_t last_ind = op_seq.size() - 1;
  const auto &last_node_ind = op_seq.operations().at(last_ind);
  const auto &last_node = operations.at(last_node_ind);
  const auto &last_node_outputs = last_node.getOutputs();
  for (const auto &op_seq_output_ind : op_seq.getOutputs())
  {
    if (last_node_outputs.contains(op_seq_output_ind) == false)
      return false;
  }

  return true;
}

ir::OperandIndex Fp32ToFp16Converter::newCopiedOperand(const ir::OperandIndex &op_ind)
{
  auto &operands = _lowered_graph.graph().operands();
  const auto &obj = operands.at(op_ind);
  auto new_op_ind = operands.emplace(obj.shape(), obj.typeInfo());
  return new_op_ind;
}

void Fp32ToFp16Converter::setNewOperandLowerInfo(const ir::OpSequenceIndex &op_seq_ind,
                                                 const ir::OperandIndex &new_op_ind)
{
  const auto &lower_info = _lowered_graph.getLowerInfo(op_seq_ind);
  assert(lower_info != nullptr);
  auto new_lower_info = std::make_unique<compiler::OperandLowerInfo>();
  auto permute_factor = compiler::PermuteFactor(lower_info->backend(), lower_info->layout());
  new_lower_info->addDefPermuteFactor(permute_factor);
  new_lower_info->addUsePermuteFactor(permute_factor);
  _lowered_graph.setLowerInfo(new_op_ind, std::move(new_lower_info));
}

void Fp32ToFp16Converter::setNewOperationLowerInfo(const ir::OpSequenceIndex &op_seq_ind,
                                                   const ir::OpSequenceIndex &new_op_seq_ind)
{
  const auto &lower_info = _lowered_graph.getLowerInfo(op_seq_ind);
  assert(lower_info != nullptr);

  auto new_lower_info =
    std::make_unique<compiler::OperationLowerInfo>(lower_info->backend(), lower_info->layout());
  _lowered_graph.setLowerInfo(new_op_seq_ind, std::move(new_lower_info));
}

void Fp32ToFp16Converter::manipulateInput(const ir::OpSequenceIndex &op_seq_ind,
                                          const ir::OperandIndex &op_seq_input_ind,
                                          const ir::OperandIndex &new_op_ind)
{
  auto &operands = _lowered_graph.graph().operands();
  auto &operations = _lowered_graph.graph().operations();

  auto &op_seq = _lowered_graph.op_seqs().at(op_seq_ind);

  auto &first_node_ind = op_seq.operations().at(0);
  auto &first_node = operations.at(first_node_ind);
  assert(first_node.getInputs().contains(op_seq_input_ind));

  auto &input_obj = operands.at(op_seq_input_ind);
  assert(input_obj.isConstant() == false);

  auto &new_op_obj = operands.at(new_op_ind);

  // The same inputs having the index as op_seq_input_ind are replaced all at once
  op_seq.replaceInputs(op_seq_input_ind, new_op_ind);
  first_node.replaceInputs(op_seq_input_ind, new_op_ind);

  // op_seq_obj doesn't have uses/def
  input_obj.removeUse(first_node_ind);
  new_op_obj.insertUse(first_node_ind);
}

void Fp32ToFp16Converter::manipulateOutput(const ir::OpSequenceIndex &op_seq_ind,
                                           const ir::OperandIndex &op_seq_output_ind,
                                           const ir::OperandIndex &new_op_ind)
{
  auto &operands = _lowered_graph.graph().operands();
  auto &operations = _lowered_graph.graph().operations();

  auto &op_seq = _lowered_graph.op_seqs().at(op_seq_ind);

  size_t last_ind = op_seq.size() - 1;
  auto &last_node_ind = op_seq.operations().at(last_ind);
  auto &last_node = operations.at(last_node_ind);
  assert(last_node.getOutputs().contains(op_seq_output_ind));

  auto &output_obj = operands.at(op_seq_output_ind);
  assert(output_obj.isConstant() == false);

  auto &new_op_obj = operands.at(new_op_ind);

  // The same outputs having the index as op_seq_output_ind are replaced all at once
  op_seq.replaceOutputs(op_seq_output_ind, new_op_ind);
  last_node.replaceOutputs(op_seq_output_ind, new_op_ind);

  // op_seq_obj doesn't have uses/def
  assert(output_obj.getDef() == last_node_ind);
  output_obj.unsetDef();
  new_op_obj.setDef(last_node_ind);
}

ir::OperationIndex
Fp32ToFp16Converter::newOperationConvertFp32ToFp16(const ir::OperandIndex &op_seq_input_ind,
                                                   const ir::OperandIndex &new_op_ind)
{
  auto &operands = _lowered_graph.graph().operands();
  auto &operations = _lowered_graph.graph().operations();

  auto &input_obj = operands.at(op_seq_input_ind);
  auto &new_op_obj = operands.at(new_op_ind);

  std::unique_ptr<ir::Operation> new_node(
    new ir::operation::ConvertFp32ToFp16({op_seq_input_ind}, {new_op_ind}));
  const auto new_node_ind = operations.push(std::move(new_node));

  input_obj.insertUse(new_node_ind);
  new_op_obj.setDef(new_node_ind);

  return new_node_ind;
}

ir::OperationIndex
Fp32ToFp16Converter::newOperationConvertFp16ToFp32(const ir::OperandIndex &op_seq_output_ind,
                                                   const ir::OperandIndex &new_op_ind)
{
  auto &operands = _lowered_graph.graph().operands();
  auto &operations = _lowered_graph.graph().operations();

  auto &output_obj = operands.at(op_seq_output_ind);
  auto &new_op_obj = operands.at(new_op_ind);

  std::unique_ptr<ir::Operation> new_node(
    new ir::operation::ConvertFp16ToFp32({new_op_ind}, {op_seq_output_ind}));
  const auto new_node_ind = operations.push(std::move(new_node));

  new_op_obj.insertUse(new_node_ind);
  output_obj.setDef(new_node_ind);

  return new_node_ind;
}

ir::OpSequenceIndex Fp32ToFp16Converter::newOpSequence(const ir::OpSequenceIndex &op_seq_ind,
                                                       const ir::OperationIndex &node_index)
{
  auto &node = _lowered_graph.graph().operations().at(node_index);
  const auto &lower_info = _lowered_graph.getLowerInfo(op_seq_ind);
  assert(lower_info != nullptr);
  auto layout = lower_info->layout();

  auto op_seq = std::make_unique<ir::OpSequence>(layout);
  op_seq->appendOperation(node_index);
  op_seq->setOutputs(node.getOutputs());
  op_seq->setInputs(node.getInputs());

  return _lowered_graph.op_seqs().emplace(std::move(op_seq));
}

// The op_seq(Fp16To32)'s output operand is the next to op_seq (Fp32To16)?
// If so, connect Fp16To32's previous OpSeq to Fp32To16's next OpSeq
//
// Assume that an OpSequence has an operation for easy explaination
//
// BEFORE)
//
// [OPERATION] // OpSeq#0
//    |
//   OP#0
//    |
// [FP16TO32]  // OpSeq#1
//    |
//   OP#1
//    |
// [FP32TO16]  // OpSeq#2
//    |
//   OP#2
//    |
// [OPERATION] // OpSeq#3
//
//
// AFTER)
//
// [OPERATION] // OpSeq#0
//    |
//   OP#0
//    |
// [OPERATION] // OpSeq#3
//
void Fp32ToFp16Converter::removeContiguousConvertOpSequences()
{
  // Prepare InputToOpSeqs map
  const auto input_to_op_seqs = prepareInputToOpSeqs();

  // Find OpSequences to delete while manipulating input of OpSeq.
  auto opseq_map_to_delete = findOpSequencesContiguous(input_to_op_seqs);

  // Find Operations to delete
  auto list_to_delete_op_seqs = getListOpSequences(opseq_map_to_delete);
  auto list_to_delete_ops = findOperationsToDelete(list_to_delete_op_seqs);

  // Before deleting, manipulateInputs of OpSeq & Operation
  manipulateContiguousOpSequences(input_to_op_seqs, opseq_map_to_delete);

  // Delete OpSequences & Operations & obj's use/def & operands
  deleteContiguousOpSequences(list_to_delete_op_seqs, list_to_delete_ops);
}

Fp32ToFp16Converter::OpSeqIndexToOpSeqIndexList
Fp32ToFp16Converter::findOpSequencesContiguous(const InputToOpSeqs &input_to_op_seqs) const
{
  const auto &op_seqs = _lowered_graph.op_seqs();
  OpSeqIndexToOpSeqIndexList opseq_map_to_delete;

  //
  // Assume that an Operation an OpSequence for easy explaination
  //
  // [OPERATION]
  //    |
  //   OP#0
  //    |
  // [FP16TO32]  // op_seq_ind_fp16_to_fp32 & op_seq_fp16_to_fp32
  //    |
  //   OP#1      // output_ind_fp16_fp32
  //    |
  // [FP32TO16]  // op_seq_ind
  //    |
  //   OP#2
  //    |
  // [OPERATION]
  //
  for (auto it = _list_fp16_to_fp32.cbegin(); it != _list_fp16_to_fp32.cend(); ++it)
  {
    // fp16_to_fp32's input/output num is always 1
    auto &op_seq_ind_fp16_to_fp32 = *it;
    auto &op_seq_fp16_to_fp32 = op_seqs.at(op_seq_ind_fp16_to_fp32);
    assert(op_seq_fp16_to_fp32.size() == 1);
    assert(op_seq_fp16_to_fp32.getInputs().size() == 1);

    auto &output_ind_fp16_to_fp32 = op_seq_fp16_to_fp32.getOutputs().at(0);
    auto found_input_in_op_seqs = input_to_op_seqs.find(output_ind_fp16_to_fp32);
    if (found_input_in_op_seqs == input_to_op_seqs.end())
    {
      continue;
    }

    // DO NOT FORGET THE CASE
    //
    //    |
    // [FP16TO32]
    //    |
    //   OP#0---------------------.
    //    |                       |
    // [FP32TO16]              [FP32TO16]
    //    |                       |
    //   OP#1                    OP#2
    //    |                       |
    // [OPERATION]             [OPERATION]
    //
    for (const auto &op_seq_ind : found_input_in_op_seqs->second)
    {
      auto found_in_fp32_to_fp16 = _list_fp32_to_fp16.find(op_seq_ind);
      if (found_in_fp32_to_fp16 != _list_fp32_to_fp16.end())
      {
        if (opseq_map_to_delete.find(op_seq_ind_fp16_to_fp32) == opseq_map_to_delete.end())
        {
          opseq_map_to_delete[op_seq_ind_fp16_to_fp32].emplace(op_seq_ind);
        }
        else
        {
          opseq_map_to_delete[op_seq_ind_fp16_to_fp32].insert(op_seq_ind);
        }

        VERBOSE(Fp32ToFp16Converter) << "Contiguous from " << op_seq_ind_fp16_to_fp32 << "(ToFp32)"
                                     << " to " << op_seq_ind << "(ToFp16)" << std::endl;
      }
    }
  }

  return opseq_map_to_delete;
}

Fp32ToFp16Converter::InputToOpSeqs Fp32ToFp16Converter::prepareInputToOpSeqs() const
{
  const auto &op_seqs = _lowered_graph.op_seqs();

  InputToOpSeqs input_to_op_seqs;
  op_seqs.iterate([&](const ir::OpSequenceIndex &op_seq_idx, const ir::OpSequence &op_seq) {
    for (auto &&input : op_seq.getInputs() | ir::Remove::UNDEFINED)
    {
      auto it = input_to_op_seqs.find(input);
      if (it == input_to_op_seqs.end())
      {
        input_to_op_seqs[input].emplace(op_seq_idx);
      }
      else
      {
        input_to_op_seqs[input].insert(op_seq_idx);
      }
    }
  });

  return input_to_op_seqs;
}

Fp32ToFp16Converter::OpSeqIndexList
Fp32ToFp16Converter::getListOpSequences(const OpSeqIndexToOpSeqIndexList &opseq_map_to_delete) const
{
  OpSeqIndexList list;
  for (const auto &it : opseq_map_to_delete)
  {
    const auto &opseq_ind_fp16_to_fp32 = it.first;
    if (list.find(opseq_ind_fp16_to_fp32) == list.end())
    {
      list.emplace(opseq_ind_fp16_to_fp32);
    }

    for (const auto &opseq_ind_fp32_to_fp16 : it.second)
    {
      if (list.find(opseq_ind_fp32_to_fp16) == list.end())
      {
        list.emplace(opseq_ind_fp32_to_fp16);
      }
    }
  }
  return list;
}

ir::OperandIndexSequence
Fp32ToFp16Converter::findOperationsToDelete(const OpSeqIndexList &list_to_delete_op_seqs) const
{
  const auto &operations = _lowered_graph.graph().operations();
  const auto &op_seqs = _lowered_graph.op_seqs();

  ir::OperandIndexSequence list_to_delete_ops;
  for (const auto &op_seq_ind : list_to_delete_op_seqs)
  {
    const auto &op_seq = op_seqs.at(op_seq_ind);
    assert(op_seq.size() == 1);

    const auto &first_node_ind = op_seq.operations().at(0);
    const auto &first_node = operations.at(first_node_ind);
    assert(first_node.opcode() == ir::OpCode::ConvertFp32ToFp16 ||
           first_node.opcode() == ir::OpCode::ConvertFp16ToFp32);

    for (const auto &ind : first_node.getOutputs())
    {
      list_to_delete_ops.append(ind);
    }
  }

  return list_to_delete_ops;
}

void Fp32ToFp16Converter::manipulateContiguousOpSequences(
  const InputToOpSeqs &input_to_op_seqs, const OpSeqIndexToOpSeqIndexList &opseq_map_to_delete)
{
  auto &op_seqs = _lowered_graph.op_seqs();

  //
  // [OPERATION]
  //    |
  //   OP#0      // input_ind_fp16_to_fp32
  //    |
  // [FP16TO32]  // op_seq_ind_fp16_to_fp32 & op_seq_fp16_to_fp32
  //    |
  //   OP#1
  //    |
  // [FP32TO16]  // op_seq_ind_fp32_to_fp16, op_seq_fp32_to_fp16
  //    |
  //   OP#2      // output_ind_fp32_to_fp16
  //    |
  // [OPERATION] // op_seq_ind_next_to_fp16
  //
  for (auto &&it : opseq_map_to_delete)
  {
    // fp16_to_fp32's input/output num is always 1
    auto &op_seq_ind_fp16_to_fp32 = it.first;
    auto &op_seq_fp16_to_fp32 = op_seqs.at(op_seq_ind_fp16_to_fp32);
    auto &input_ind_fp16_to_fp32 = op_seq_fp16_to_fp32.getInputs().at(0);

    for (const auto &op_seq_ind_fp32_to_fp16 : it.second)
    {
      auto &op_seq_fp32_to_fp16 = op_seqs.at(op_seq_ind_fp32_to_fp16);
      assert(op_seq_fp32_to_fp16.size() == 1);
      assert(op_seq_fp32_to_fp16.getInputs().size() == 1);

      auto &output_ind_fp32_to_fp16 = op_seq_fp32_to_fp16.getOutputs().at(0);
      auto found_next_to_fp16 = input_to_op_seqs.find(output_ind_fp32_to_fp16);
      assert(found_next_to_fp16 != input_to_op_seqs.end());

      for (const auto &op_seq_ind_next_to_fp16 : found_next_to_fp16->second)
      {
        manipulateInput(op_seq_ind_next_to_fp16, output_ind_fp32_to_fp16, input_ind_fp16_to_fp32);
      }
      //
      // [OPERATION]
      //    |
      //   OP#0      // input_ind_fp16_to_fp32
      //    |
      // [OPERATION] // op_seq_ind_next_to_fp16
      //
    }
  }
}

void Fp32ToFp16Converter::deleteContiguousOpSequences(
  const OpSeqIndexList &list_to_delete_op_seqs, const ir::OperandIndexSequence &list_to_delete_ops)
{
  auto &operands = _lowered_graph.graph().operands();
  auto &operations = _lowered_graph.graph().operations();
  auto &op_seqs = _lowered_graph.op_seqs();

  for (const auto &op_seq_ind : list_to_delete_op_seqs)
  {
    auto &op_seq = op_seqs.at(op_seq_ind);
    assert(op_seq.size() == 1);
    VERBOSE(Fp32ToFp16Converter) << "Delete OpSeq " << op_seq_ind << std::endl;

    auto &first_node_ind = op_seq.operations().at(0);
    auto &first_node = operations.at(first_node_ind);
    assert(first_node.opcode() == ir::OpCode::ConvertFp32ToFp16 ||
           first_node.opcode() == ir::OpCode::ConvertFp16ToFp32);
    VERBOSE(Fp32ToFp16Converter) << "Delete Node " << first_node_ind << std::endl;

    // Uses
    for (const auto &ind : first_node.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      auto &obj = operands.at(ind);
      obj.removeUse(first_node_ind);
      VERBOSE(Fp32ToFp16Converter)
        << "Operand " << ind << "'s Use(Node" << first_node_ind << ") is removed" << std::endl;
    }

    // Def
    for (const auto &ind : first_node.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      auto &obj = operands.at(ind);
      assert(obj.getDef() == first_node_ind);
      obj.unsetDef();
      VERBOSE(Fp32ToFp16Converter)
        << "Operand " << ind << "'s Def(Node" << first_node_ind << ") is removed" << std::endl;
    }

    // Operation
    operations.remove(first_node_ind);
    VERBOSE(Fp32ToFp16Converter) << "Node" << first_node_ind << " is removed" << std::endl;

    // OpSequence
    op_seqs.remove(op_seq_ind);
    VERBOSE(Fp32ToFp16Converter) << "OpSeq" << op_seq_ind << " is removed" << std::endl;
  }

  // Operand
  for (const auto &ind : list_to_delete_ops)
  {
    operands.remove(ind);
    VERBOSE(Fp32ToFp16Converter) << "Operand " << ind << " is removed" << std::endl;
  }
}

} // namespace compiler

} // namespace onert

#endif
