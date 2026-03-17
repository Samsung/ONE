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

#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include "Convert.h"
#include "TensorBags.h"

#include <coco/IR/Data.h>
#include <coco/IR/Module.h>

#include <schema_generated.h>

#include <map>

using namespace nncc::core::ADT;

namespace tflimport
{

/**
 * @brief Extracts and holds operand(tensor) information such as name, shape, and type
 */
class TensorContext
{
public:
  void prepare(const tflite::SubGraph *graph);

  const std::string &name(uint32_t tensor_id) { return _name_ctx[tensor_id]; }
  const tensor::Shape &shape(uint32_t tensor_id) { return _shape_ctx[tensor_id]; }
  const tflite::TensorType &type(uint32_t tensor_id) { return _type_ctx[tensor_id]; }

private:
  std::map<uint32_t, std::string> _name_ctx;
  std::map<uint32_t, tensor::Shape> _shape_ctx;
  std::map<uint32_t, tflite::TensorType> _type_ctx;
};

/**
 * @brief Class that holds operator codes and related methods
 */
class TflOpCodeContext
{
public:
  TflOpCodeContext(const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>> *opcodes);

  /**
   * @brief Returns BuiltinOperator value of the operator
   */
  tflite::BuiltinOperator builtin_code(const tflite::Operator *op) const;

  /**
   * @brief Returns human readable name of the operator code of the operator
   *
   * @note TF lite InterpreterBuilder sets an error state and returns error code
   *       for invalid opcode. Here we just return human readable message as
   *       this method returns a name for the operator code.
   */
  std::string opcode_name(const tflite::Operator *op) const;

public:
  static bool is_valid(const tflite::OperatorCode *opcode);
  static bool is_custom(const tflite::OperatorCode *opcode);

private:
  std::vector<const tflite::OperatorCode *> _opcodes;
};

/**
 * @brief Class to read and provide buffer information of tflite
 */
class TflBufferContext
{
public:
  template <typename T> struct TflBuffer
  {
    TflBuffer(const T *p, size_t s) : ptr{p}, len{s} {};
    const T *ptr;
    size_t len;
  };

public:
  explicit TflBufferContext(const tflite::Model *tfl_model);

public:
  template <typename T>
  TflBuffer<T> tensor_buffer(const tflite::SubGraph *graph, uint32_t tensor_idx) const
  {
    TflBufferContext::TflBuffer<T> res{nullptr, 0};
    const auto *tensor = graph->tensors()->Get(tensor_idx);
    uint32_t tfl_buf_id = tensor->buffer();

    assert(_buffer_ctx.size() > tfl_buf_id);

    const tflite::Buffer *tfl_buffer = _buffer_ctx.at(tfl_buf_id);

    if (auto *array = tfl_buffer->data())
    {
      if (size_t size = array->size())
      {
        assert(size % sizeof(T) == 0);

        res.len = size / sizeof(T);
        res.ptr = reinterpret_cast<const T *>(array->data());
      }
    }

    return res;
  }

private:
  std::map<uint32_t /* Buffer ID */, const tflite::Buffer *> _buffer_ctx;
};

/**
 * @brief Class to store context to build IR from tflite
 */
class GraphBuilderContext
{
public:
  explicit GraphBuilderContext(coco::Module *m, coco::Data *d, coco::Block *block,
                               TensorBags &tensor_bags, TensorContext &tensor_context,
                               TflBufferContext &buffer_context, const tflite::SubGraph *graph)
    : _m(m), _d(d), _block(block), _tensor_bags(tensor_bags), _tensor_context(tensor_context),
      _buffer_context(buffer_context), _graph(graph)
  {
    // DO NOTHING
  }

  GraphBuilderContext() = delete;
  GraphBuilderContext(const GraphBuilderContext &) = delete;
  GraphBuilderContext(GraphBuilderContext &&) = delete;

public:
  coco::Module *m() { return _m; }
  coco::Data *d() { return _d; }
  coco::Block *block() { return _block; }
  TensorContext &tensor() { return _tensor_context; }
  TensorBags &bags() { return _tensor_bags; }
  TflBufferContext &buffer() { return _buffer_context; }
  const tflite::SubGraph *graph() { return _graph; }

private:
  coco::Module *_m;
  coco::Data *_d;
  coco::Block *_block;
  TensorContext &_tensor_context;
  TensorBags &_tensor_bags;
  TflBufferContext &_buffer_context;
  const tflite::SubGraph *_graph;
};

} // namespace tflimport

#endif // __CONTEXT_H__
