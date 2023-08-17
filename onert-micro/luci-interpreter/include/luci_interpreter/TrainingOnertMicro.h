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

#ifndef LUCI_INTERPRETER_TRAINING_ONERT_MICRO_H
#define LUCI_INTERPRETER_TRAINING_ONERT_MICRO_H

#include "luci_interpreter/TrainingSettings.h"
#include "luci_interpreter/Interpreter.h"
#include "core/TrainingModule.h"

namespace luci_interpreter
{
namespace training
{

class TrainingOnertMicro
{
public:
  explicit TrainingOnertMicro(Interpreter *interpreter, TrainingSettings &settings);

  ~TrainingOnertMicro();

  Status enableTrainingMode();

  Status disableTrainingMode(bool resetWeights = false);

  Status train(uint32_t number_of_train_samples, const uint8_t *train_data,
               const uint8_t *label_train_data);

  Status test(uint32_t number_of_train_samples, const uint8_t *test_data,
              const uint8_t *label_test_data, void *metric_value_result);

private:
  Interpreter *_interpreter;

  TrainingSettings &_settings;

  TrainingModule _module;

  bool _is_training_mode;
};

} // namespace training
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_TRAINING_ONERT_MICRO_H

#endif // ENABLE_TRAINING
