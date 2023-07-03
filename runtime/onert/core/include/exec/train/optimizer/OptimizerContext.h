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

#ifndef __ONERT_EXEC_TRAIN_OPTIMIZER_RUN_CONTEXT_H__
#define __ONERT_EXEC_TRAIN_OPTIMIZER_RUN_CONTEXT_H__

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

/**
 * @class   Opimizer Context class for all optimizers
 * @brief   Optimizer Context
 */
class OptimizerContext
{
public:
  /**
   * @brief Construct a new optimizer context object
   *
   */
  OptimizerContext(const backend::IPortableTensor &grad,
                   backend::train::ITrainableTensor &trainable, size_t iter = 0)
    : _gradient{grad}, _trainable{trainable}, iteration(iter)
  {
  }

  /**
   * @brief Get the trainable tensor object
   *
   * @return ITrainableTensor Reference to the trainable tensor
   */
  backend::train::ITrainableTensor &getTrainableTensor() { return _trainable; }

  /**
   * @brief Get the gradient tensor object
   *
   * @return ITensor Reference to the grad tensor
   */
  const backend::IPortableTensor &getGradientTensor() const { return _gradient; }

  /**
   * @brief Get the current iteration value
   *
   * @return size_t The number of training steps
   */
  size_t getIteration() const { return iteration; }

private:
  const backend::IPortableTensor &_gradient;    /**< gradient tensor to be used by optimizer */
  backend::train::ITrainableTensor &_trainable; /**< trainable tensor to be updated by optimizer */
  size_t iteration;                             /**< iteration number */
};

} // namespace optimizer
} // namespace train
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_OPTIMIZER_RUN_CONTEXT_H__
