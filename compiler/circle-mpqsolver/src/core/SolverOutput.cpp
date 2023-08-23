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

#include "SolverOutput.h"

#include <iostream>

SolverOutput &SolverOutput::get(void)
{
  static SolverOutput d;
  return d;
}

const SolverOutput &SolverOutput::operator<<(const std::string &message) const
{
  if (_turn_on)
  {
    std::cout << message;
  }

  return *this;
}

const SolverOutput &SolverOutput::operator<<(float value) const
{
  if (_turn_on)
  {
    std::cout << value;
  }

  return *this;
}

void SolverOutput::TurnOn(bool on) { _turn_on = on; }
