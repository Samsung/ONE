/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __RECORD_MINMAX_RECORD_FUNCTION_H__
#define __RECORD_MINMAX_RECORD_FUNCTION_H__

#include <vector>
#include <cstdint>

namespace record_minmax
{

/**
 * @brief  getNthPercentile calculates the n-th percentile of input vector (0.0 <= n <= 100.0)
 *         linear interpolation is used when the desired percentile lies between two data points
 */
float getNthPercentile(std::vector<float> &vector, float percentile);

/**
 * @brief  getMovingAverage calculates the weighted moving average of input vector
 *         The initial value is the minimum (or maximum) value of the first batch of the vector
 */
float getMovingAverage(const std::vector<float> &vector, const float alpha,
                       const uint8_t batch_size, bool is_min);

} // namespace record_minmax

#endif // __RECORD_MINMAX_RECORD_FUNCTION_H__
