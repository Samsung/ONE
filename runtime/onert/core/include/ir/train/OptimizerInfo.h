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

#ifndef __ONERT_IR_TRAIN_OPTIMIZER_INFO_H__
#define __ONERT_IR_TRAIN_OPTIMIZER_INFO_H__

#include "OptimizerCode.h"

namespace onert
{
namespace ir
{
namespace train
{

struct OptimizerOption
{
  virtual ~OptimizerOption(){};

  virtual OptimizerOption *clone() const = 0;
};

struct SGDOption final : public OptimizerOption
{
  float learning_rate;

  ~SGDOption(){};

  SGDOption *clone() const { return new SGDOption(*this); }
};

struct AGDOption final : public OptimizerOption
{
  float learning_rate;
  float beta_1;
  float beta_2;
  float epsilon;

  ~AGDOption(){};

  AGDOption *clone() const { return new AGDOption(*this); }
};

struct OptimizerInfo
{
  OptimizerCode optim_code;
  // Q: Is this value updatable?
  // Q: Is it possible to use shared_ptr here?
  // This is not-yet fully used
  std::unique_ptr<OptimizerOption> optim_option;

  float learning_rate; // TODO deperecate
  // TODO Add properties

  OptimizerInfo() : optim_option(nullptr){};

  OptimizerInfo(const OptimizerInfo &another)
  {
    optim_code = another.optim_code;
    optim_option = std::unique_ptr<OptimizerOption>(another.optim_option->clone());
    learning_rate = another.learning_rate;
  }

  OptimizerInfo &operator=(const OptimizerInfo &another)
  {
    optim_code = another.optim_code;
    optim_option = std::unique_ptr<OptimizerOption>(another.optim_option->clone());
    learning_rate = another.learning_rate;
    return *this;
  }
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_OPTIMIZER_INFO_H__
