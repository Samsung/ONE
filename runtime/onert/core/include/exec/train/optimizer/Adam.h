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

#ifndef __ONERT_EXEC_TRAIN_OPTIMIZER_ADAM_H__
#define __ONERT_EXEC_TRAIN_OPTIMIZER_ADAM_H__

#include "exec/train/optimizer/Optimizer.h"

#include "backend/basic/Tensor.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace exec
{
namespace train
{
namespace optimizer
{

/**
 * @class   Adam optimizer class
 * @brief   Adam optimizer
 */
class Adam : public Optimizer
{
public:
  struct Property
  {
    double beta1{0.9};
    double beta2{0.999};
    double epsilon{1e-07};
  };

public:
  explicit Adam() : _props{}, _learning_rate{0.001} {}
  explicit Adam(const Property &props) : _props{props}, _learning_rate{0.001} {}
  explicit Adam(double lr) : _props{}, _learning_rate{lr} {}
  explicit Adam(const Property &props, double lr) : _props{props}, _learning_rate{lr} {}

public:
  /**
   * @brief Get the name of optimizer
   *
   * @return The name of optimizer
   */
  std::string name() const override { return std::string{"Adam"}; }

  /**
   * @brief Get the Learning Rate
   *
   * @param iteration The number of training steps
   * @return Learning rate
   */
  double getLearningRate(uint32_t iteration) const override;

  /**
   * @brief Apply gradient to a trainable tensor
   *
   * @param factors UpdateFactors to be used for applying gradient to a trainable tensor
   */
  void applyGradient(const UpdateFactors &factors) const override;

  // TODO Add methods for exporting optimizer information

private:
  Property _props;
  double _learning_rate;
};

} // namespace optimizer
} // namespace train
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_OPTIMIZER_ADAM_H__
