/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _NNC_SOFT_BACKEND_SEQUENCED_IR_H_
#define _NNC_SOFT_BACKEND_SEQUENCED_IR_H_

#include "mir/Shape.h"
#include "mir/Operation.h"

#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <list>

namespace nnc
{

namespace sir
{

const size_t INVALID_TENSOR_ID = std::numeric_limits<size_t>::max();

/**
 * @brief Represents variable used in artifact.
 * This variable can store inputs, outputs of network and temporary data.
 */
struct TensorDescriptor
{
  /**
   * input      tensors of this type supposed to be set outside of artifact
   * persistent tensors store data after inference process is over, this include NN outputs
   * temporary  tensors are not accessible outside artifact in any way,
   *            they are created and destructed on demand
   */
  enum class Type
  {
    input,
    persistent,
    temporary
  };

  size_t id;
  Type type;
  std::string name;
  // if _shape.rank() == 0 - assume shape is not known for this tensor on compilation
  mir::Shape shape;
};

/**
 * @brief Action represents operation in inference sequence that is needed to
 */
struct Action
{

  /**
   * Defines which type of action to perform
   * createTmp responsible for creation of temporary tensor in inference sequence
   * destroyTmp responsible for deletion of temporary tensor
   * transpose
   */
  enum class Type
  {
    createTmp,
    destroyTmp,
    callFunction,
    transposeTensor
  };

  explicit Action(Type t) : type(t) {}

  virtual ~Action() = default;

  Type type;
};

struct TransposeTensor : public Action
{

  TransposeTensor(size_t input, size_t output, std::vector<int32_t> &&perm)
    : Action(Type::transposeTensor), perm(std::move(perm)), input(input), output(output)
  {
  }

  std::vector<int32_t> perm;
  size_t input;
  size_t output;
};

struct CreateTmp : public Action
{

  explicit CreateTmp(size_t tid) : Action(Type::createTmp), tensorId(tid) {}

  size_t tensorId;
};

struct DestroyTmp : public Action
{

  explicit DestroyTmp(size_t tid) : Action(Type::destroyTmp), tensorId(tid) {}

  size_t tensorId;
};

struct CallFunction : public Action
{

  CallFunction(mir::Operation *op, std::string func_name, std::vector<size_t> &&inputs,
               std::vector<size_t> &&outputs)
    : Action(Type::callFunction), mirOp(op), funcName(std::move(func_name)), inputs(inputs),
      outputs(outputs), paramStartOffset(0)
  {
  }

  CallFunction() : Action(Type::callFunction), mirOp(nullptr), paramStartOffset(0) {}

  mir::Operation *mirOp;
  std::string funcName;
  // list of input tensors
  std::vector<size_t> inputs;
  // list of output tensors
  std::vector<size_t> outputs;
  size_t paramStartOffset;
};

} // namespace sir

} // namespace nnc

#endif // _NNC_SOFT_BACKEND_SEQUENCED_IR_H_
