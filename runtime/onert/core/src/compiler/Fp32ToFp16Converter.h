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

#ifndef __ONERT_COMPILER_FP32_TO_FP16_CONVERTER_H__
#define __ONERT_COMPILER_FP32_TO_FP16_CONVERTER_H__

#include "compiler/LoweredGraph.h"

namespace onert
{

namespace compiler
{

class Fp32ToFp16Converter
{
public:
  Fp32ToFp16Converter(compiler::LoweredGraph &lowered_graph);

public:
  void run();

private:
  using OpSeqIndexList = std::unordered_set<ir::OpSequenceIndex>;
  using InputToOpSeqs = std::unordered_map<ir::OperandIndex, OpSeqIndexList>;
  using OpSeqIndexToOpSeqIndexList = std::unordered_map<ir::OpSequenceIndex, OpSeqIndexList>;

private:
  void appendOpSequences();
  void optimize();
  void convertOperands();
  void convertDatas();
  void printOpSequences(const std::string &pre_msg = std::string(),
                        const std::string &post_msg = std::string());

  bool checkOperandType(const ir::OperandIndex &op_ind) const;
  bool checkOperandsOfOpSequence(const ir::OpSequence &op_seq) const;

  void appendNewOpSeqForConvertFp32ToFp16(const ir::OpSequenceIndex &op_seq_ind,
                                          ir::OpSequence &op_seq);
  void appendNewOpSeqForConvertFp16ToFp32(const ir::OpSequenceIndex &op_seq_ind,
                                          ir::OpSequence &op_seq);

  ir::OperandIndex newCopiedOperand(const ir::OperandIndex &op_ind);
  ir::OperationIndex newOperationConvertFp32ToFp16(const ir::OperandIndex &op_seq_input_ind,
                                                   const ir::OperandIndex &new_op_ind);
  ir::OperationIndex newOperationConvertFp16ToFp32(const ir::OperandIndex &op_seq_output_ind,
                                                   const ir::OperandIndex &new_op_ind);
  ir::OpSequenceIndex newOpSequence(const ir::OpSequenceIndex &op_seq_ind,
                                    const ir::OperationIndex &node_index);

  void setNewOperandLowerInfo(const ir::OpSequenceIndex &op_seq_ind,
                              const ir::OperandIndex &new_op_ind);
  void setNewOperationLowerInfo(const ir::OpSequenceIndex &op_seq_ind,
                                const ir::OpSequenceIndex &new_op_seq_ind);

  void manipulateInput(const ir::OpSequenceIndex &op_seq_ind,
                       const ir::OperandIndex &op_seq_input_ind,
                       const ir::OperandIndex &new_op_ind);
  void manipulateOutput(const ir::OpSequenceIndex &op_seq_ind,
                        const ir::OperandIndex &op_seq_output_ind,
                        const ir::OperandIndex &new_op_ind);

  void removeContiguousConvertOpSequences();
  InputToOpSeqs prepareInputToOpSeqs() const;
  OpSeqIndexToOpSeqIndexList
  findOpSequencesContiguous(const InputToOpSeqs &intput_to_op_seqs) const;
  OpSeqIndexList getListOpSequences(const OpSeqIndexToOpSeqIndexList &opseq_map_to_delete) const;
  ir::OperandIndexSequence
  findOperationsToDelete(const OpSeqIndexList &list_to_delete_op_seqs) const;
  void manipulateContiguousOpSequences(const InputToOpSeqs &input_to_op_seqs,
                                       const OpSeqIndexToOpSeqIndexList &opseq_map_to_delete);
  void deleteContiguousOpSequences(const OpSeqIndexList &list_to_delete_op_seqs,
                                   const ir::OperandIndexSequence &list_to_delete_ops);

  void convertOperandsOfOpSequence(ir::OpSequence &op_seq);

private:
  compiler::LoweredGraph &_lowered_graph;
  OpSeqIndexList _list_fp32_to_fp16;
  OpSeqIndexList _list_fp16_to_fp32;
};

} // namespace compiler

} // namespace onert

#endif // __ONERT_COMPILER_FP32_TO_FP16_CONVERTER_H__

#endif
