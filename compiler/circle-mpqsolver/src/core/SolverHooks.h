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

#ifndef __MPQSOLVER_SOLVER_HOOKS_H__
#define __MPQSOLVER_SOLVER_HOOKS_H__

#include <luci/IR/Module.h>

#include "core/Quantizer.h"

#include <string>

namespace mpqsolver
{
namespace core
{

class SolverHooks
{
public:
  /**
   * @brief called on the start of iterative search
   * @param model_path path of original float model to quantize
   * @param q8error error of Q8 quantization
   * @param q16error error of Q16 quantization
   */
  virtual void onBeginSolver(const std::string &model_path, float q8error, float q16error) = 0;

  /**
   * @brief called on the start of current iteration
   */
  virtual void onBeginIteration() = 0;

  /**
   * @brief called at the end of current iteration
   * @param layers model nodes with specific quantization parameters
   * @param def_dtype default quantization dtype
   * @param error error of quantization for current iteration
   */
  virtual void onEndIteration(const LayerParams &layers, const std::string &def_dtype,
                              float error) const = 0;

  /**
   * @brief called at the end of iterative search
   * @param layers model nodes with specific quantization parameters
   * @param def_dtype default quantization dtype
   * @param qerror final error of quantization
   */
  virtual void onEndSolver(const LayerParams &layers, const std::string &def_dtype,
                           float qerror) = 0;
};

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_SOLVER_HOOKS_H__
