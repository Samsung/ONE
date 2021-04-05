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

#include "PartitionExport.h"
#include "HelperPath.h"

#include <crew/PConfig.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace
{

std::string export_file_path(const std::string &output_base, const std::string &input,
                             const std::string &ext)
{
  auto filename_ext = partee::get_filename_ext(input);
  auto pos = filename_ext.find_last_of(".");
  assert(pos > 0);
  auto filename = filename_ext.substr(0, pos);
  auto filepath = output_base + "/" + filename + ".conn" + ext;
  return filepath;
}

} // namespace

namespace
{

void graph_io_to_config_part(loco::Graph *graph, crew::Part &part)
{
  assert(graph != nullptr);

  auto *gis = graph->inputs();
  auto *gos = graph->outputs();
  for (uint32_t i = 0; i < gis->size(); ++i)
  {
    auto *gi = gis->at(i);
    assert(gi != nullptr);
    part.inputs.push_back(gi->name());
  }
  for (uint32_t i = 0; i < gos->size(); ++i)
  {
    auto *go = gos->at(i);
    assert(go != nullptr);
    part.outputs.push_back(go->name());
  }
}

void pms2config(const luci::PartedModules &pms, crew::PConfig &pconfig)
{
  for (auto &pmodule : pms.pmodules)
  {
    auto *graph = pmodule.module->graph();

    crew::Part part;
    part.model_file = pmodule.name;
    graph_io_to_config_part(graph, part);

    pconfig.parts.push_back(part);
  }
}

} // namespace

namespace partee
{

bool export_part_conn_json(const std::string &output_base, const std::string &input,
                           const luci::Module *source, luci::PartedModules &pms)
{
  crew::PConfig pconfig;

  // TODO is graph I/O using main graph is enough?
  auto *graph = source->graph();

  pconfig.source.model_file = input;
  graph_io_to_config_part(graph, pconfig.source);

  pms2config(pms, pconfig);

  auto filepath_json = export_file_path(output_base, input, ".json");
  std::ofstream fs(filepath_json.c_str(), std::ofstream::binary | std::ofstream::trunc);
  if (not fs.good())
  {
    std::cerr << "ERROR: Failed to create file: " << filepath_json;
    return false;
  }
  if (not write_json(fs, pconfig))
  {
    std::cerr << "ERROR: Failed to write json file: " << filepath_json;
    return false;
  }
  fs.close();

  return true;
}

bool export_part_conn_ini(const std::string &output_base, const std::string &input,
                          const luci::Module *source, luci::PartedModules &pms)
{
  crew::PConfig pconfig;

  // TODO is graph I/O using main graph is enough?
  auto *graph = source->graph();

  pconfig.source.model_file = input;
  graph_io_to_config_part(graph, pconfig.source);

  pms2config(pms, pconfig);

  auto filepath_ini = export_file_path(output_base, input, ".ini");
  std::ofstream fs(filepath_ini.c_str(), std::ofstream::binary | std::ofstream::trunc);
  if (not fs.good())
  {
    std::cerr << "ERROR: Failed to create file: " << filepath_ini;
    return false;
  }
  if (not write_ini(fs, pconfig))
  {
    std::cerr << "ERROR: Failed to write ini file: " << filepath_ini;
    return false;
  }
  fs.close();

  return true;
}

} // namespace partee
