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

/**
 * @file        arm_compute.h
 * @brief       This file contains arm_compute library related classes
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __INTERNAL_ARM_COMPUTE_H__
#define __INTERNAL_ARM_COMPUTE_H__

#include <arm_compute/core/ITensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/Tensor.h>

namespace internal
{
namespace arm_compute
{
namespace operand
{

/**
 * @brief Class to access the tensor object
 */
class Object
{
public:
  Object() = default;

public:
  Object(const std::shared_ptr<::arm_compute::ITensor> &tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get the tensor pointer
   * @return The tensor pointer
   */
  ::arm_compute::ITensor *ptr(void) const { return _tensor.get(); }

private:
  std::shared_ptr<::arm_compute::ITensor> _tensor;

public:
  /**
   * @brief Access the tensor object and run the given function
   : @param[in] fn The actual behavior when accessing the tensor object
   * @return N/A
   */
  void access(const std::function<void(::arm_compute::ITensor &tensor)> &fn) const;
};

} // namespace operand
} // namepsace arm_compute
} // namespace internal

#include "internal/Model.h"

#include <map>

namespace internal
{
namespace arm_compute
{
namespace operand
{

/**
 * @brief Class to manage Object instances
 */
class Context
{
public:
  /**
   * @brief Set index and tensor pair
   * @param[in] ind The operand index
   * @param[in] tensor The tensor object
   * @return This object reference
   */
  Context &set(const ::internal::tflite::operand::Index &ind,
               const std::shared_ptr<::arm_compute::ITensor> &tensor);

public:
  /**
   * @brief Check if the tensor for given index is exist
   * @param[in] ind The operand Index
   * @return @c true if the entry for ind is exist, otherwise @c false
   */
  bool exist(const ::internal::tflite::operand::Index &ind) const
  {
    return _objects.find(ind.asInt()) != _objects.end();
  }

public:
  /**
   * @brief Lookup the tensor with the given index
   * @param[in] ind The index as the key
   * @return The object const reference
   */
  const Object &at(const ::internal::tflite::operand::Index &ind) const
  {
    return _objects.at(ind.asInt());
  }

  /**
   * @brief Lookup the tensor with the given index
   * @param[in] ind The index as the key
   * @return The object reference
   */
  Object &at(const ::internal::tflite::operand::Index &ind) { return _objects.at(ind.asInt()); }

private:
  std::map<int, Object> _objects;
};

} // namespace operand
} // namepsace arm_compute
} // namespace internal

#include <arm_compute/runtime/IFunction.h>

namespace internal
{
namespace arm_compute
{
namespace op
{

/**
 * @brief Class to wrap IFunction
 */
class Step
{
public:
  /**
   * @brief Construct a Step object
   * @param[in] func The compiled code to be executed
   */
  Step(std::unique_ptr<::arm_compute::IFunction> &&func) : _func{std::move(func)}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Run _func
   * @return N/A
   */
  void run(void) const { _func->run(); }

public:
  /**
   * @brief Get member @c _name
   * @return The name as const reference
   */
  const std::string &name(void) const { return _name; }
  /**
   * @brief Get member @c _name
   * @return The name as reference
   */
  std::string &name(void) { return _name; }

private:
  std::string _name;
  std::unique_ptr<::arm_compute::IFunction> _func;
#ifdef TFLITE_PROFILING_ENABLED
public:
  /**
   * @brief Get member @c _op_index
   * @return The operation index as value
   */
  int op_idx() const { return _op_idx; }
  /**
   * @brief Get member @c _op_index
   * @return The operation index as reference
   */
  int &op_idx() { return _op_idx; }
private:
  int _op_idx;
#endif
};

} // namespace op
} // namepsace arm_compute
} // namespace internal

namespace internal
{
namespace arm_compute
{
namespace op
{

/**
 * @brief Class managing compiled operation code Sequence
 */
class Sequence
{
public:
  /**
   * @brief Get size of sequence
   * @return Number of sequence steps
   */
  uint32_t size(void) const { return _functions.size(); }

public:
  /**
   * @brief Append a Function to the sequence
   * @param[in] func Function to be appended
   * @return This object reference
   */
  Sequence &append(std::unique_ptr<::arm_compute::IFunction> &&func)
  {
    _functions.emplace_back(std::move(func));
    return (*this);
  }

public:
  /**
   * @brief Get the step entry on the index @c n
   * @param[in] n The index
   * @return The step object as reference
   */
  Step &at(uint32_t n) { return _functions.at(n); }
  /**
   * @brief Get the step entry on the index @c n
   * @param[in] n The index
   * @return The step object as const reference
   */
  const Step &at(uint32_t n) const { return _functions.at(n); }

private:
  // TODO Rename _functions as _steps
  std::vector<Step> _functions;
};

} // namespace op
} // namepsace arm_compute
} // namespace internal

namespace internal
{
namespace arm_compute
{

/**
 * @brief Class to manage compiled operation sequence
 */
class Plan
{
public:
  /**
   * @brief Construct a Plan object
   * @param[in] model Model that we want to compile
   */
  Plan(const std::shared_ptr<const ::internal::tflite::Model> &model) : _model(model)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get the model object
   * @return The model object as const reference
   */
  const ::internal::tflite::Model &model(void) const { return *_model; }

public:
  /**
   * @brief Get operand context
   * @return The operand context as reference
   */
  operand::Context &operands(void) { return _operands; }
  /**
   * @brief Get operand context
   * @return The operand context as const reference
   */
  const operand::Context &operands(void) const { return _operands; }

public:
  /**
   * @brief Get operation sequence
   * @return The operation sequence as reference
   */
  op::Sequence &operations(void) { return _ops; }
  /**
   * @brief Get operation sequence
   * @return The operation sequence as const reference
   */
  const op::Sequence &operations(void) const { return _ops; }

private:
  std::shared_ptr<const ::internal::tflite::Model> _model;
  operand::Context _operands;
  op::Sequence _ops;
};

} // namepsace arm_compute
} // namespace internal

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{

/**
 * @brief Check if this runtime runs on GPU or NEON
 * @return @c true if GPU mode, otherwise @c false
 */
bool isGpuMode();

#define CAST_CL(tensor) static_cast<::arm_compute::CLTensor *>(tensor)
#define CAST_NE(tensor) static_cast<::arm_compute::Tensor *>(tensor)

} // namepsace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_H__
