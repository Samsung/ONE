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

#ifndef __NNFW_API_TEST_CIRCLE_GEN_H__
#define __NNFW_API_TEST_CIRCLE_GEN_H__

#include <circle_schema_generated.h>

#include <vector>

/**
 * @brief Class for storing flatbuffer buffer
 *
 * This is a simple wrapper for a finished FlatBufferBuilder. It owns the buffer and a user can
 * get the buffer pointer and size.
 */
class CircleBuffer
{
public:
  CircleBuffer() = default;
  explicit CircleBuffer(flatbuffers::FlatBufferBuilder &&fbb) : _fbb{std::move(fbb)}
  {
    _fbb.Finished(); // The build must have been finished, so check that here
  }

  uint8_t *buffer() { return _fbb.GetBufferPointer(); }
  size_t size() { return _fbb.GetSize(); }

private:
  flatbuffers::FlatBufferBuilder _fbb;
};

/**
 * @brief Circle flatbuffer file generator
 *
 * This is a helper class for generating circle file.
 *
 */
class CircleGen
{
public:
  struct TensorParams
  {
    std::vector<int32_t> shape;
    circle::TensorType tensor_type = circle::TensorType::TensorType_FLOAT32;
    uint32_t buffer = 0;
    std::string name;
  };

  struct OperatorParams
  {
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    int version = 1;
  };

public:
  CircleGen();

  template <typename T> uint32_t addBuffer(const std::vector<T> &buf_vec)
  {
    auto buf = reinterpret_cast<const uint8_t *>(buf_vec.data());
    auto size = buf_vec.size() * sizeof(T);
    return addBuffer(buf, size);
  }
  uint32_t addBuffer(const uint8_t *buf, size_t size);
  uint32_t addTensor(const TensorParams &params);
  uint32_t setInputsAndOutputs(const std::vector<int> &inputs, const std::vector<int> &outputs);
  CircleBuffer finish();

  // ===== Add Operator methods begin =====

  uint32_t addOperatorAdd(const OperatorParams &params, circle::ActivationFunctionType actfn);
  uint32_t addOperatorAveragePool2D(const OperatorParams &params, circle::Padding padding,
                                    int stride_w, int stride_h, int filter_w, int filter_h,
                                    circle::ActivationFunctionType actfn);

  // NOTE Please add addOperator functions ABOVE this lie
  // ===== Add Operator methods end =====

private:
  uint32_t addOperatorWithOptions(const OperatorParams &params, circle::BuiltinOperator opcode,
                                  circle::BuiltinOptions options_type,
                                  flatbuffers::Offset<void> options);
  uint32_t addOperatorCode(circle::BuiltinOperator opcode);
  flatbuffers::Offset<circle::Buffer> buildBuffer(const uint8_t *buf, size_t size);
  flatbuffers::Offset<circle::Tensor> buildTensor(const TensorParams &params);
  flatbuffers::Offset<circle::SubGraph> buildSubGraph();

private:
  flatbuffers::FlatBufferBuilder _fbb{1024};
  std::vector<flatbuffers::Offset<circle::Buffer>> _buffers;
  std::vector<flatbuffers::Offset<circle::OperatorCode>> _opcodes;

  // per-subgraph
  std::vector<int> _inputs;
  std::vector<int> _outputs;
  std::vector<flatbuffers::Offset<circle::Tensor>> _tensors;
  std::vector<flatbuffers::Offset<circle::Operator>> _operators;
};

#endif // __NNFW_API_TEST_CIRCLE_GEN_H__
