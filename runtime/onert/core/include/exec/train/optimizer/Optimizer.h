/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_TRAIN_OPTIMIZER_OPTIMIZER_H__
#define __ONERT_EXEC_TRAIN_OPTIMIZER_OPTIMIZER_H__

#include "backend/IPortableTensor.h"
#include "backend/train/ITrainableTensor.h"

#include <string>

namespace onert
{
namespace exec
{
namespace train
{
namespace optimizer
{

// Gradient tensor, Trainable Tensor, Number of training steps
using UpdateFactors =
  std::tuple<const backend::IPortableTensor &, backend::train::ITrainableTensor &, size_t>;

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Base class for all optimizers
 */
class Optimizer
{
public:
  virtual ~Optimizer() = default;

  /**
   * @brief Get the name of optimizer
   *
   * @return The name of optimizer
   */
  virtual std::string name() const { return std::string{"Invalid"}; }

  /**
   * @brief Get the Learning Rate
   *
   * @param iteration The number of training steps
   * @return Learning rate
   */
  virtual double getLearningRate(uint32_t iteration) const = 0;

  /**
   * @brief Apply gradient to a trainable tensor
   *
   * @param factors UpdateFactors to be used for applying gradient to a trainable tensor
   */
  virtual void applyGradient(const UpdateFactors &factors) const = 0;

  // TODO Add member functions for exporting optimizer information
};

} // namespace optimizer
} // namespace train
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_OPTIMIZER_OPTIMIZER_H__
