/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ReadModule.h"

#include <luci/Pass/ShapeInferencePass.h>
#include <luci/Pass/TypeInferencePass.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>
#include <luci/Service/Validate.h>

#include <logo/Phase.h>

#include <iostream>
#include <string>
#include <vector>

std::unique_ptr<luci::Module> ReadModule(std::string &input_path)
{
  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data = file_loader.load();
  const circle::Model *circle_model = circle::GetModel(model_data.data());
  if (circle_model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << input_path << "'" << std::endl;
    return nullptr;
  }

  luci::Importer importer;
  auto module = importer.importModule(circle_model);
  assert(module->size() > 0);

  for (size_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);
    if (graph == nullptr)
      return nullptr;

    {
      logo::Phase phase;

      phase.emplace_back(std::make_unique<luci::ShapeInferencePass>());
      phase.emplace_back(std::make_unique<luci::TypeInferencePass>());
      phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
      phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

      logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{graph};
      phase_runner.run(phase);
    }

    if (!luci::validate(graph))
      return nullptr;
  }
  return module;
}
