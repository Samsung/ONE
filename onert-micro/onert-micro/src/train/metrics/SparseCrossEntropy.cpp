
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

#include "train/metrics/SparseCrossEntropy.h"

#include <cmath>
#include <algorithm>

using namespace onert_micro;
using namespace onert_micro::train;
using namespace onert_micro::train::metrics;

/*
 * Y - calculated value
 * Y_i - probability of target class (Correct)
 * E(Y, Y_i) = -log(Y_i))
 */
float SparseCrossEntropy::calculateValue(const uint32_t flat_size, const float *calculated_data,
                                   const float *target_data)
{
  float result_value = 0.f;

  // Sparse Cross Entropy uses target data as a integer label of target class.
  uint32_t label_index = static_cast<uint32_t>(target_data[0]);
  result_value = std::log(calculated_data[label_index] + float(10.0e-32));
  return -result_value;
}
