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

#ifndef _NNC_SOFT_BACKEND_SERIALIZER_H_
#define _NNC_SOFT_BACKEND_SERIALIZER_H_

#include "mir/Visitor.h"
#include "mir/Shape.h"
#include "mir/TensorVariant.h"
#include "ModelAnalyzer.h"

#include <vector>
#include <cstdint>

namespace nnc
{

/**
 * @brief Serializer of network parameters for soft backend
 *
 * Serializer class responsible for serialization of given computational graph parameters and
 * binding of inference operations to this data.
 * It owns buffer that contains serialized data.
 * To serialize data `serialize` method should be called with sequence from ModelAnalyzer object
 * To gather this vector use `getBuffer` method.
 * Objects of this class are one-off and not designed to serialize more than one IR
 */
class Serializer : public mir::Visitor
{
public:
  void visit(mir::ops::AbsOp &op) override;
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

  void serialize(std::vector<std::unique_ptr<sir::Action>> &inference_sequence);

  const std::vector<char> &getBuffer() const { return _buffer; }

  uint32_t getFormatVersion() const { return _formatVersion; }

  uint32_t getModelHash() const { return _modelHash; }

protected:
  void visit_fallback(mir::Operation &op) override;

private:
  /**
   * @brief Low level function to serialize untyped data buffer
   * @param data Buffer containing data to serialize
   * @param size Size of data to serialize
   */
  void packData(const void *data, size_t size);
  /**
   * @brief Serialize trivially copyable objects
   * @tparam T Type of object to serialize
   * @param obj Reference to object to serialize
   */
  template <typename T> void serializeT(const T &obj);
  /**
   * @brief Serialize Tensor shape object
   * @param s shape to serialize
   */
  void serializeShape(const mir::Shape &s);
  /**
   * @brief Function serializes type of given tensor base data,
   * it's shape and raw data in 'c' format(i.e. layout of multidimensional C array)
   * @param t Tensor to serialize
   */
  void serializeTensor(const mir::TensorVariant &t);
  /**
   * @brief Serialize strides.
   * @param strides The strides to serialize.
   */
  void serializeStrides(const std::vector<std::int32_t> &strides);
  /**
   * @brief Serialize pads for operations like Conv2D
   * @tparam Op Operation type
   * @param op Reference to operation where pads are stored
   * @param padsRank Number of pads to serialize
   */
  template <class Op> void serializePads(const Op &op, int32_t number_of_pads);

  sir::CallFunction *_curOp = nullptr;
  const uint32_t _formatVersion = 1;
  uint32_t _modelHash = 0;
  std::vector<char> _buffer;
};

} // namespace nnc

#endif //_NNC_SOFT_BACKEND_SERIALIZER_H_
