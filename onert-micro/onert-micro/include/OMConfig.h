/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_CONFIG_H
#define ONERT_MICRO_CONFIG_H

#include <stdint.h>

namespace onert_micro
{

enum OMOptimizationStrategy
{
  SGD,
  RMSProp,
  ADAM,
};

// Training specific options
struct OMTrainingConfig
{
  float lambda = 0.f;
  float beta_squares = 0.9f;
  float beta = 0.9f;
  float epsilon = 10e-8;
  uint16_t batches = 1;
  OMOptimizationStrategy optimization_strategy = SGD;
};

struct OMConfig
{
  bool keep_input = false;
  bool cmsis_nn = false;
  // For case with divided weights and circle file
  char *wof_ptr = nullptr;
  bool train_mode = false;
  // Training specific options
  OMTrainingConfig train_config;
};

} // namespace onert_micro

#endif // ONERT_MICRO_CONFIG_H
