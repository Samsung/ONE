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

#ifndef __MPQSOLVER_CORE_EVALUATOR_H__
#define __MPQSOLVER_CORE_EVALUATOR_H__

#include "ErrorMetric.h"

#include <luci/IR/Module.h>
#include <luci/CircleQuantizer.h>

#include <string>
#include <vector>

namespace mpqsolver
{
namespace core
{

class DatasetEvaluator final
{
public:
  /**
   * @brief create Evaluator for comparing output of ref_module on h5file
   */
  DatasetEvaluator(const luci::Module *ref_module, const std::string &h5file,
                   const ErrorMetric &metric);
  DatasetEvaluator() = delete;
  ~DatasetEvaluator() = default;

  /**
   * @brief evaluate trgt_fq_module (fake-quantized)
   * returns error-metric
   */
  float evaluate(const luci::Module *trgt_fq_module) const;

private:
  /**
   * @brief throws if there is something wrong with the module
   */
  void validate(const luci::Module *module) const;

private:
  const luci::Module *_ref_module = nullptr;
  std::string _h5file;
  WholeOutput _ref_output;
  const ErrorMetric *_metric = nullptr;
};

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_CORE_EVALUATOR_H__
