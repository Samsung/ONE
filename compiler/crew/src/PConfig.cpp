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

namespace
{

void part_to_section_io(const crew::Part &part, crew::Section &section)
{
  uint32_t idx = 1;
  for (auto &input : part.inputs)
  {
    std::string key = "i" + std::to_string(idx);
    section.items.emplace(key, input);
    idx++;
  }
  idx = 1;
  for (auto &output : part.outputs)
  {
    std::string key = "o" + std::to_string(idx);
    section.items.emplace(key, output);
    idx++;
  }
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

bool write_ini(std::ostream &os, const PConfig &pconfig)
{
  crew::Sections sections;

  // make [source]
  crew::Section section_source;
  section_source.name = "source";
  section_source.items["file"] = pconfig.source.model_file;
  part_to_section_io(pconfig.source, section_source);
  sections.push_back(section_source);

  // make [models]
  crew::Section section_models;
  section_models.name = "models";
  uint32_t idx = 1;
  for (auto &part : pconfig.parts)
  {
    std::string key = "m" + std::to_string(idx);
    section_models.items[key] = part.model_file;
    idx++;
  }
  sections.push_back(section_models);

  for (auto &part : pconfig.parts)
  {
    // make circle model section
    crew::Section section_model;
    section_model.name = part.model_file;
    section_model.items["file"] = part.model_file;
    part_to_section_io(part, section_model);
    sections.push_back(section_model);
  }

  write_ini(os, sections);

  return true;
}

} // namespace crew
