/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifdef ENABLE_TRAINING

#ifndef LUCI_INTERPRETER_TRAINING_SETTINGS_H
#define LUCI_INTERPRETER_TRAINING_SETTINGS_H

#include <stdint.h>

namespace luci_interpreter
{

namespace training
{

enum Status
{
  Ok,
  Error,
  EnableTrainModeError,
  DoubleTrainModeError
};

enum MetricsTypeEnum
{
  MSE,
  MAE
};

enum LossTypeEnum
{
  MSE_Loss
};

enum OptimizerTypeEnum
{
  SGD
};

struct TrainingSettings
{
  MetricsTypeEnum metric = MSE;
  LossTypeEnum error_type = MSE_Loss;
  OptimizerTypeEnum optimizer_type = SGD;
  uint32_t number_of_epochs = 1;
  uint32_t batch_size = 1;
  float learning_rate = 0.00001;
  uint32_t number_of_last_trainable_layers = 1;
};

} // namespace training
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_TRAINING_SETTINGS_H

#endif // ENABLE_TRAINING
