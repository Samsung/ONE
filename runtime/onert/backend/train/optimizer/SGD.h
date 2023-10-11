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

#ifndef __ONERT_BACKEND_TRAIN_OPTIMIZER_SGD_H__
#define __ONERT_BACKEND_TRAIN_OPTIMIZER_SGD_H__

#include <exec/train/optimizer/Optimizer.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace optimizer
{

/**
 * @class   SGD optimizer class
 * @brief   SGD optimizer
 */
class SGD : public exec::train::optimizer::Optimizer
{
public:
  using UpdateFactors = exec::train::optimizer::UpdateFactors;

public:
  struct Property
  {
    double momentum{0.0};
    bool nesterov{false};
  };

public:
  explicit SGD() : _props{}, _learning_rate{0.01} {}
  explicit SGD(const Property &props) : _props{props}, _learning_rate{0.01} {}
  explicit SGD(double lr) : _props{}, _learning_rate{lr} {}
  explicit SGD(const Property &props, double lr) : _props{props}, _learning_rate{lr} {}

public:
  /**
   * @brief Get the name of optimizer
   *
   * @return The name of optimizer
   */
  std::string name() const override { return std::string{"SGD"}; }

  /**
   * @brief Get the Learning Rate
   *
   * @param iteration The number of training steps
   * @return Learning rate
   */
  double getLearningRate(uint32_t iteration = 0) const override;

  /**
   * @brief Get the number of optimizer variables
   *s
   * @return The number of optimizer variables
   */
  virtual uint32_t getVarCount() const override { return 0; };

  /**
   * @brief Apply gradient to a trainable tensor
   *
   * @param factors UpdateFactors to be used for applying gradient to a trainable tensor
   */
  void applyGradient(const UpdateFactors &factors) const override;

private:
  Property _props;
  double _learning_rate;
};

} // namespace optimizer
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPTIMIZER_SGD_H__
