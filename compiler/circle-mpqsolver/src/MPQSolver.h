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

#ifndef __MPQSOLVER_MPQSOLVER_SOLVER_H__
#define __MPQSOLVER_MPQSOLVER_SOLVER_H__

#include "core/Quantizer.h"
#include "core/DumpingHooks.h"

#include <luci/IR/CircleNodes.h>

#include <memory>
#include <string>

namespace mpqsolver
{

class MPQSolver
{
public:
  MPQSolver(const core::Quantizer::Context &ctx);

  virtual ~MPQSolver() = default;

  /**
   * @brief run solver for recorded float module at module_path
   */
  virtual std::unique_ptr<luci::Module> run(const std::string &module_path) = 0;

  /**
   * @brief set all intermediate artifacts to be saved
   */
  void setSaveIntermediate(const std::string &save_path);

protected:
  std::unique_ptr<luci::Module> readModule(const std::string &path);

protected:
  std::string _input_quantization;
  std::string _output_quantization;
  std::unique_ptr<core::Quantizer> _quantizer;
  std::unique_ptr<core::DumpingHooks> _hooks;
};

} // namespace mpqsolver

#endif //__MPQSOLVER_MPQSOLVER_SOLVER_H__
