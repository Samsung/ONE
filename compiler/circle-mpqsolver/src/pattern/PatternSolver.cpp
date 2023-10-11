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

#include "PatternSolver.h"

#include <iostream>

using namespace mpqsolver::pattern;

PatternSolver::PatternSolver(const std::string &input_quantization,
                             const std::string &output_quantization,
                             const std::vector<QuantizationPattern> &patterns)
  : MPQSolver("", 1.f, input_quantization, output_quantization)
{
  MPQOptions options{patterns};
  set_mpq_options(options);
}

std::unique_ptr<luci::Module> PatternSolver::run(const std::string &module_path)
{
  auto module = read_module(module_path);

  resolve_patterns(module.get());

  auto layer_params = get_frozen_params();

  if (!_quantizer->quantize(module.get(), "uint8", layer_params))
  {
    std::cerr << "ERROR: Failed to quantize model" << std::endl;
    return nullptr;
  }

  return module;
}
