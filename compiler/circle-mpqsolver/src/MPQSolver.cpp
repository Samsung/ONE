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

#include "MPQSolver.h"

#include <luci/ImporterEx.h>
#include <iostream>

using namespace mpqsolver;

MPQSolver::MPQSolver(const std::string &input_data_path, float qerror_ratio,
                     const std::string &input_quantization, const std::string &output_quantization)
  : _input_data_path(input_data_path), _qerror_ratio(qerror_ratio),
    _input_quantization(input_quantization), _output_quantization(output_quantization)
{
  _quantizer = std::make_unique<core::Quantizer>(_input_quantization, _output_quantization);
}

void MPQSolver::set_save_intermediate(const std::string &save_path)
{
  _hooks = std::make_unique<core::DumpingHooks>(save_path);
}

std::unique_ptr<luci::Module> MPQSolver::read_module(const std::string &path)
{
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(path);
  if (module.get() == nullptr)
  {
    throw std::runtime_error("Failed to load model");
  }

  return module;
}
