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

#include "crew/PConfig.h"
#include "crew/PConfigIni.h"

#include "PConfigJson.h"

#include <utility>

namespace
{

bool read_part(const crew::Section &section, crew::Part &part)
{
  // construct Source from section_source
  part.model_file = crew::find(section, "file");
  if (part.model_file.empty())
    return false;

  // read inputs for Source
  for (int32_t i = 1;; ++i)
  {
    std::string item = "i" + std::to_string(i);
    std::string input = crew::find(section, item);
    if (input.empty())
      break;

    part.inputs.push_back(input);
  }
  // read outputs for Source
  for (int32_t i = 1;; ++i)
  {
    std::string item = "o" + std::to_string(i);
    std::string output = crew::find(section, item);
    if (output.empty())
      break;

    part.outputs.push_back(output);
  }
  return true;
}

} // namespace

namespace crew
{

bool read_ini(const std::string &path, PConfig &pconfig)
{
  auto sections = crew::read_ini(path);

  auto section_source = crew::find(sections, "source");
  auto section_models = crew::find(sections, "models");
  if (section_source.name != "source" || section_models.name != "models")
  {
    return false;
  }

  if (!read_part(section_source, pconfig.source))
  {
    return false;
  }

  // get models list
  std::vector<std::string> models;
  for (int32_t i = 1;; ++i)
  {
    std::string item = "m" + std::to_string(i);
    std::string model = crew::find(section_models, item);
    if (model.empty())
      break;

    models.push_back(model);
  }

  for (auto &model : models)
  {
    auto section_model = crew::find(sections, model);

    Part part;
    if (!read_part(section_model, part))
    {
      return false;
    }
    pconfig.parts.push_back(part);
  }

  return true;
}

} // namespace crew
