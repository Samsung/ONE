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

#ifndef _MIR_OPERATION_H_
#define _MIR_OPERATION_H_

#include "mir/TensorType.h"

#include <deque>
#include <string>
#include <limits>
#include <vector>

namespace mir
{

class IVisitor;

class Operation
{
public:
  enum class Type
  {
#define HANDLE_OP(OpType, OpClass) OpType,
#include "mir/Operations.inc"
#undef HANDLE_OP
  };

  /// @brief Represents a use of an operation output.
  struct Use
  {
    Use(Operation *node, std::size_t index) : _node(node), _index(index) {}

    Operation *getNode() const { return _node; }

    std::size_t getIndex() const { return _index; }

    bool operator==(const Use &other) { return _node == other._node && _index == other._index; }

  private:
    Operation *_node;
    std::size_t _index;
  };

  /// @brief Represents an output of a node.
  class Output
  {
  public:
    Output(Operation *node, std::size_t index) : _node(node), _index(index) {}

    ~Output() = default;

    Output(const Output &) = delete;
    Output(Output &&) = delete;
    Output &operator=(const Output &) = delete;
    Output &operator=(Output &&) = delete;

    /// @brief Returns the node this is an output of.
    Operation *getNode() { return _node; }
    const Operation *getNode() const { return _node; }

    /// @brief Returns the index of this output among all the outputs of the node.
    std::size_t getIndex() const { return _index; }

    /// @brief Returns the inputs that consume this output.
    const std::vector<Use> &getUses() const { return _uses; }

    /// @brief Adds the specified use to the uses of this output.
    void addUse(Use use) { _uses.push_back(use); }

    /// @brief Removes the specified use from the uses of this output.
    void removeUse(Use use);

    /// @brief Replace the defs of all uses of this output with the specified def.
    void replaceAllUsesWith(Output *new_def);

    /// @brief Gets the type of this output.
    const TensorType &getType() const { return _type; }

    /// @brief Sets the type of this output.
    /// @warning Use with caution, because it can make the IR inconsistent.
    void setType(const TensorType &type) { _type = type; }

    // Convenient accessors.
    DataType getElementType() const { return getType().getElementType(); }
    const Shape &getShape() const { return getType().getShape(); }

    // TODO Remove in favor of `setType`.
    void setShape(const Shape &shape) { setType(TensorType(_type.getElementType(), shape)); }

    const std::string &getName() const { return _name; }
    void setName(const std::string &name) { _name = name; }

    /// @brief Set AffineQuantization to Ouput
    void setQuantization(const mir::AffineQuantization &quant)
    {
      setType(TensorType(getElementType(), getShape(), quant));
    }

  private:
    Operation *_node;
    std::size_t _index;
    std::vector<Use> _uses;
    TensorType _type;
    std::string _name;
  };

  virtual ~Operation() = default;

  Type getType() const { return _type; }

  std::size_t getId() const { return _id; }
  void setId(std::size_t id) { _id = id; }

  std::size_t getNumInputs() const { return _inputs.size(); }
  std::size_t getNumOutputs() const { return _outputs.size(); }

  std::deque<Output *> &getInputs() { return _inputs; }
  const std::deque<Output *> &getInputs() const { return _inputs; }

  std::deque<Output> &getOutputs() { return _outputs; }
  const std::deque<Output> &getOutputs() const { return _outputs; }

  Output *getInput(std::size_t index)
  {
    assert(index < _inputs.size());
    return _inputs[index];
  }

  const Output *getInput(std::size_t index) const
  {
    assert(index < _inputs.size());
    return _inputs[index];
  }

  Output *getOutput(std::size_t index)
  {
    assert(index < _outputs.size());
    return &_outputs[index];
  }

  const Output *getOutput(std::size_t index) const
  {
    assert(index < _outputs.size());
    return &_outputs[index];
  }

  const Shape &getInputShape(std::size_t index) const { return getInput(index)->getShape(); }

  const Shape &getOutputShape(std::size_t index) const { return getOutput(index)->getShape(); }

  void accept(IVisitor *v);

  virtual Operation *copyWithInputs(const std::vector<Output *> &inputs) = 0;

protected:
  Operation(Type type, const std::vector<Output *> &inputs, std::size_t num_outputs = 1);

  void setOutputType(std::size_t index, const TensorType &type) { getOutput(index)->setType(type); }

private:
  Type _type;
  std::size_t _id = std::numeric_limits<std::size_t>::max();
  std::deque<Output *> _inputs;
  std::deque<Output> _outputs;
};

/**
 * @return the opcode of operation in string format, like "Add", "Conv2d", etc.
 */
const std::string &getTypeName(Operation::Type type);

} // namespace mir

#endif //_MIR_OPERATION_H_
