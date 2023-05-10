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

#ifndef __MPQSOLVER_MPQSOLEVR_SOLVER_H__
#define __MPQSOLVER_MPQSOLEVR_SOLVER_H__

#include <core/DumpingHooks.h>

#include <memory>
#include <string>

namespace mpqsolver
{

class MPQSolver
{
public:
  /**
   * @brief construct Solver using input_data_path for .h5 file,
   * qerror_ratio to set target qerror, and input_quantization/output_quantization to set
   * quantization type at input/output respectively
   */
  MPQSolver(const std::string &input_data_path, float qerror_ratio,
            const std::string &input_quantization, const std::string &output_quantization);
  virtual ~MPQSolver() = default;

  /**
   * @brief run solver for recorded float module at module_path
   */
  virtual std::unique_ptr<luci::Module> run(const std::string &module_path) = 0;

  /**
   * @brief set all intermediate artifacts to be saved
   */
  void set_save_intermediate(const std::string &save_path);

protected:
  std::string _input_data_path;
  std::string _input_quantization;
  std::string _output_quantization;
  float _qerror_ratio = 0.f; // quantization error ratio
  std::unique_ptr<core::DumpingHooks> _hooks;
};

} // namespace mpqsolver

#endif //__MPQSOLVER_MPQSOLEVR_SOLVER_H__
