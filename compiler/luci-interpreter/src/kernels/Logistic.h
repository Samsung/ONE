/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELS_LOGISTIC_H
#define LUCI_INTERPRETER_KERNELS_LOGISTIC_H

#include "core/Kernel.h"

namespace luci_interpreter
{
namespace kernels
{

class Logistic : public Kernel
{
public:
  Logistic(const Tensor *input, Tensor *output);

  std::vector<const Tensor *> getInputTensors() const override { return {_input}; }
  std::vector<Tensor *> getOutputTensors() const override { return {_output}; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;
  void evalQuantized() const;
  void populateLookupTable();
  void setTableValue(uint8_t value, uint8_t idx) { _table[idx] = value; };
  uint8_t getTableValue(uint8_t idx) const { return _table[idx]; };

private:
  const Tensor *const _input;
  Tensor *const _output;
  uint8_t _table[256];
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_LOGISTIC_H
