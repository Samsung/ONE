/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file  Interpreter.h
 * @brief This file contains Interpreter class for interpretation
 */
#ifndef __ONERT_INTERP_INTERPRETER_H__
#define __ONERT_INTERP_INTERPRETER_H__

#include "ExecEnv.h"

namespace onert
{
namespace interp
{

/**
 * @brief Class for interpretation
 */
class Interpreter
{

public:
  /**
   * @brief Construct a new Interpreter object (deleted)
   */
  Interpreter() = delete;
  /**
   * @brief     Construct a new Interpreter object
   * @param[in] env Execution environment variable for interpreter object
   */
  Interpreter(std::unique_ptr<ExecEnv> env) : _env{std::move(env)}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Run interpreter until there is no operation to execute
   */
  void run();

private:
  std::unique_ptr<ExecEnv> _env;
};

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_INTERPRETER_H__
