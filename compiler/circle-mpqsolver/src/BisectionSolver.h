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

#include <luci/IR/Module.h>

#include <memory>
#include <string>

namespace mpqsolver
{

class BisectionSolver final
{
public:
  struct Options
  {
    enum class Q16AtInput
    {
      Auto,
      True,
      False,
    };
    virtual ~Options() = default;

    virtual void enable(Q16AtInput) = 0;
    virtual bool query(Q16AtInput) = 0;
  };

public:
  Options *options(void);

public:
  BisectionSolver(const std::string &input_data_path, float qerror_ratio);
  ~BisectionSolver() = default;

  std::unique_ptr<luci::Module> run(const luci::Module *in_module);

private:
  std::string _input_data_path;
  float _qerror = 0.f;       // quantization error
  float _qerror_ratio = 0.f; // quantization error ratio
  std::unique_ptr<Options> _options;
};

} // namespace mpqsolver

#endif //__MPQSOLVER_BISECTION_SOLVER_H__
