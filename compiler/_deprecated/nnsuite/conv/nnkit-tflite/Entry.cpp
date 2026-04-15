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

#include "ConvBackend.h"

#include <nnsuite/conv/RandomModel.h>

#include <nnkit/Backend.h>
#include <nnkit/CmdlineArguments.h>

#include <memory>
#include <chrono>
#include <iostream>

extern "C" std::unique_ptr<nnkit::Backend> make_backend(const nnkit::CmdlineArguments &args)
{
  // Set random seed
  int32_t seed = std::chrono::system_clock::now().time_since_epoch().count();

  if (args.size() > 0)
  {
    seed = std::stoi(args.at(0), nullptr, 0);
  }

  std::cout << "SEED: " << seed << std::endl;

  const nnsuite::conv::RandomModel model{seed};

  return std::make_unique<ConvBackend>(model);
}
