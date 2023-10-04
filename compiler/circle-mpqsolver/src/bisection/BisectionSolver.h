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

#ifndef __MPQSOLVER_BISECTION_SOLVER_H__
#define __MPQSOLVER_BISECTION_SOLVER_H__

#include <core/Evaluator.h>
#include <MPQSolver.h>

#include <luci/IR/Module.h>

#include <memory>
#include <string>

namespace mpqsolver
{
namespace bisection
{

class BisectionSolver final : public MPQSolver
{
public:
  /**
   * @brief Algorithm options for running bisection algorithm
   */
  enum Algorithm
  {
    Auto,
    ForceQ16Front,
    ForceQ16Back,
  };

public:
  /**
   * @brief construct Solver using input_data_path for .h5 file,
   * qerror_ratio to set target qerror, and input_quantization/output_quantization to set
   * quantization type at input/output respectively
   */
  BisectionSolver(const std::string &input_data_path, float qerror_ratio,
                  const std::string &input_quantization, const std::string &output_quantization);
  BisectionSolver() = delete;

  /**
   * @brief run bisection for recorded float module at module_path
   */
  std::unique_ptr<luci::Module> run(const std::string &module_path) override;

  /**
   * @brief set used algorithm
   */
  void algorithm(Algorithm algorithm);

  /**
   * @brief   set visq_file path to be used in 'auto' mode
   * @details this is used to handle which way (8 or 16bit) of
   *          splitting the neural network will be the best for accuracy.
   */
  void setVisqPath(const std::string &visq_path);

private:
  float evaluate(const core::DatasetEvaluator &evaluator, const std::string &module_path,
                 const std::string &def_quant, core::LayerParams &layers);

private:
  float _qerror = 0.f; // quantization error
  Algorithm _algorithm = Algorithm::ForceQ16Front;
  std::string _visq_data_path;
};

} // namespace bisection
} // namespace mpqsolver

#endif //__MPQSOLVER_BISECTION_SOLVER_H__
