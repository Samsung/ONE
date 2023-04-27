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

#ifndef __MPQSOLVER_DUMPING_HOOKS_H__
#define __MPQSOLVER_DUMPING_HOOKS_H__

#include <luci/IR/Module.h>

#include <core/Quantizer.h>
#include <core/SolverHooks.h>
#include <core/Dumper.h>

#include <string>

namespace mpqsolver
{
namespace core
{

/**
 * @brief DumpingHooks is intended to save intermediate results
 */
class DumpingHooks final : public QuantizerHook, public SolverHooks
{
public:
  /**
   * @brief DumpingHooks constructor
   * @param save_path directory where all intermediate data will be saved
   */
  DumpingHooks(const std::string &save_path);

  /**
   * @brief called on successfull quantization
   */
  virtual void on_quantized(luci::Module *module) const override;

  /**
   * @brief called on the start of iterative search
   */
  virtual void on_begin_solver(const std::string &model_path, float q8error,
                               float q16error) override;

  /**
   * @brief called on the start of current iteration
   */
  virtual void on_begin_iteration() override;

  /**
   * @brief called at the end of current iteration
   */
  virtual void on_end_iteration(const LayerParams &layers, const std::string &def_dtype,
                                float error) const override;

  /**
   * @brief called at the end of iterative search
   */
  virtual void on_end_solver(const LayerParams &layers, const std::string &def_dtype) override;

protected:
  std::string _model_path;
  std::string _save_path;
  Dumper _dumper;
  uint32_t _num_of_iterations = 0;
  bool _in_iterations = false;
};

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_DUMPING_HOOKS_H__
