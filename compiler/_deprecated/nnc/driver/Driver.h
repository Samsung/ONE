/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNCC_DRIVER_H
#define NNCC_DRIVER_H

#include <exception>
#include <string>

#include "pass/PassManager.h"

namespace nnc
{

/**
 * @brief exceptions description class for compiler driver
 */
class DriverException : public std::exception
{
public:
  DriverException() = default;
  explicit DriverException(std::string reason) : _msg(std::move(reason)) {}
  explicit DriverException(const char *msg) : _msg(msg) {}

  const char *what() const noexcept override { return _msg.c_str(); }

private:
  std::string _msg;
};

/**
 * @brief Compiler Driver manages the whole pipeline compilation process
 */
class Driver
{
public:
  /**
   * @brief main method to run compiler driver
   * @throw DriverException if errors occurred in driver
   *        PassException if errors occurred in passes
   */
  void runDriver();

private:
  void registerBackendSpecificPasses();
  void registerOptimizationPass();
  void runPasses();

  PassManager _passManager;
};

} // namespace nnc

#endif // NNCC_DRIVER_H
