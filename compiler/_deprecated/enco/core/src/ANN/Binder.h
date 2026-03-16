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

#ifndef __ANN_BINDER_H__
#define __ANN_BINDER_H__

#include "ANN/IR/Module.h"

#include <coco/IR.h>

#include <morph/nnapi.h>

#include <type_traits>

/**
 * @brief A bridge between ann::Module and coco::Block
 */
class ANNBinder
{
public:
  ANNBinder(coco::Block *block, std::unique_ptr<ann::Module> &&module)
    : _block{block}, _module{std::move(module)}
  {
    // DO NOTHING
  }

public:
  const coco::Block *block(void) const { return _block; }
  coco::Block *block(void) { return _block; }

public:
  const ann::Module *module(void) const { return _module.get(); }

public:
  /**
   * @brief Return the set of bags that the current ANN subnet accesses
   */
  std::set<coco::Bag *> bags(void) const
  {
    std::set<coco::Bag *> res;

    for (auto it = _operands.begin(); it != _operands.end(); ++it)
    {
      res.insert(it->first);
    }

    return res;
  }

public:
  template <typename T> ann::OperandID addOperand(void)
  {
    return _module->operand()->create(ann::dtype<T>());
  };

  template <typename T> ann::OperandID addOperand(const nncc::core::ADT::tensor::Shape &shape)
  {
    return _module->operand()->create(ann::dtype<T>(), shape);
  }

public:
  template <typename T> ann::OperandID addOperand(const coco::FeatureObject *obj)
  {
    auto bag = obj->bag();
    assert(bag != nullptr);

    auto it = _operands.find(bag);

    if (it != _operands.end())
    {
      return it->second;
    }

    auto operand = addOperand<T>(morph::nnapi::as_tensor_shape(obj->shape()));
    _operands[obj->bag()] = operand;
    return operand;
  };

  template <typename T> ann::OperandID addOperand(const coco::KernelObject *obj)
  {
    auto bag = obj->bag();
    assert(bag != nullptr);

    auto it = _operands.find(bag);

    if (it != _operands.end())
    {
      return it->second;
    }

    auto operand = addOperand<T>(morph::nnapi::as_tensor_shape(obj->shape()));
    _operands[obj->bag()] = operand;
    return operand;
  };

public:
  /// @brief Set scalar weight
  template <typename T> void setOperand(const ann::OperandID &id, const T &value)
  {
    static_assert(std::is_arithmetic<T>::value, "T should be arithmetic");
    auto weight = _module->weight()->create();
    weight->fill(value);
    _module->operand()->at(id)->weight(weight);
  }

  /// @brief Set non-scalar weight
  template <typename It> void setOperand(const ann::OperandID &id, It beg, It end)
  {
    auto weight = _module->weight()->create();
    weight->fill(beg, end);
    _module->operand()->at(id)->weight(weight);
  }

public:
  void addOperation(ann::Operation::Code code, std::initializer_list<ann::OperandID> inputs,
                    std::initializer_list<ann::OperandID> outputs)
  {
    _module->operation()->create(code, inputs, outputs);
  }

public:
  /**
   * @brief Identify a sequence of coco::Bag * as subnet's inputs
   *
   * NOTE 1. This method takes input iterator over coco::Bag * values
   * NOTE 2. All the identifyInputs class except the last one will be ignored if there are
   *         multiple identifyInputs calls
   */
  template <typename It> void identifyInputs(It beg, It end)
  {
    _inputs.clear();
    _module->input()->clear();

    for (auto it = beg; it != end; ++it)
    {
      auto const bag = *it;
      _inputs.emplace_back(*it);
      _module->input()->emplace_back(_operands.at(bag));
    }
  }

  template <typename T> void identifyInputs(T &&values)
  {
    identifyInputs(std::begin(values), std::end(values));
  }

public:
  /**
   * @brief Identify a sequence of coco::Bag * as subnet's outputs
   *
   * NOTE 1. This method takes input iterator over coco::Bag * values
   * NOTE 2. All the identifyOutputs class except the last one will be ignored if there are
   *         multiple identifyOutputs calls
   */
  template <typename It> void identifyOutputs(It beg, It end)
  {
    _outputs.clear();
    _module->output()->clear();

    for (auto it = beg; it != end; ++it)
    {
      auto const bag = *it;
      _outputs.emplace_back(bag);
      _module->output()->emplace_back(_operands.at(bag));
    }
  }

  template <typename T> void identifyOutputs(T &&values)
  {
    identifyOutputs(std::begin(values), std::end(values));
  }

public:
  coco::Bag *input(uint32_t n) const { return _inputs.at(n); }
  coco::Bag *output(uint32_t n) const { return _outputs.at(n); }

public:
  /**
   * @brief Return true if a given bag has an associated operand in ANN IR
   */
  bool associated(coco::Bag *b) const { return _operands.find(b) != _operands.end(); }

  /**
   * @brief Return operand ID associated with a given bag
   * @note The behavior of operand(b) is defined only when associated(b) holds.
   */
  ann::OperandID operand(coco::Bag *b) const
  {
    assert(associated(b));
    return _operands.at(b);
  }

private:
  coco::Block *const _block;
  std::unique_ptr<ann::Module> _module;

private:
  std::vector<coco::Bag *> _inputs;
  std::vector<coco::Bag *> _outputs;

private:
  /// @brief Operand ID assigned for each coco::Bag
  std::map<coco::Bag *, ann::OperandID> _operands;
};

#endif // __ANN_BINDER_H__
