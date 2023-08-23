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

#ifndef __MPQSOLVER_SOLVER_OUTPUT_H__
#define __MPQSOLVER_SOLVER_OUTPUT_H__

#include <string>

/**
 * @brief SolverOutput prints important performance information
 */
class SolverOutput
{
private:
  /**
   * @brief construct SolverOutput
   */
  SolverOutput() = default;

public:
  /**
   * @brief get singleton object
   */
  static SolverOutput &get(void);

  /**
   * @brief print string message
   */
  const SolverOutput &operator<<(const std::string &message) const;

  /**
   * @brief print float value
   */
  const SolverOutput &operator<<(float value) const;

  /**
   * @brief turn on/off actual output
   */
  void TurnOn(bool on);

private:
  bool _turn_on = true;
};

#endif // __MPQSOLVER_SOLVER_OUTPUT_H__
