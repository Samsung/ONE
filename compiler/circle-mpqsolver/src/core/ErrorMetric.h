/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MPQSOLVER_CORE_ERROR_METRIC_H__
#define __MPQSOLVER_CORE_ERROR_METRIC_H__

#include <vector>

namespace mpqsolver
{
namespace core
{

using Buffer = std::vector<char>;
using Output = std::vector<Buffer>;
using WholeOutput = std::vector<Output>;

class ErrorMetric
{
public:
  virtual ~ErrorMetric() = default;

  /**
   * @brief abstract method for comparing first and second operands
   */
  virtual float compute(const WholeOutput &first, const WholeOutput &second) const = 0;
};

// Mean Absolute Error
class MAEMetric final : public ErrorMetric
{
public:
  /**
   * @brief compare first and second operands in MAE (Mean Average Error metric)
   */
  float compute(const WholeOutput &first, const WholeOutput &second) const;
};

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_CORE_ERROR_METRIC_H__
