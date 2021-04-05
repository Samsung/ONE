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

#include "PartitionRead.h"
#include "HelperPath.h"
#include "HelperStrings.h"

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/Log.h>

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>

namespace
{

const char *opt_bks = "--backends";
const char *opt_def = "--default";
const char *opt_part = "partition";
const char *opt_input = "input";
const char *opt_work = "work";

void print_version(void)
{
  std::cout << "circle-partitioner version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

void build_arser(arser::Arser &arser)
{
  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  arser.add_argument(opt_bks)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Backends in CSV to use for partitioning");

  arser.add_argument(opt_def)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Default backend to assign");

  arser.add_argument(opt_part)
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Partition file which provides backend to assign");
  arser.add_argument(opt_input)
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Input circle model filename");
  arser.add_argument(opt_work)
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Work folder of partition, input files exist and output files are produced");
}

std::unique_ptr<luci::Module> load_model(const std::string &input_path)
{
  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data = file_loader.load();

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<uint8_t *>(model_data.data()), model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cerr << "ERROR: Invalid input file '" << input_path << "'" << std::endl;
    return nullptr;
  }

  const circle::Model *circle_model = circle::GetModel(model_data.data());
  if (circle_model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << input_path << "'" << std::endl;
    return nullptr;
  }

  // Import from input Circle file
  luci::Importer importer;
  return importer.importModule(circle_model);
}

bool validate_partition(luci::PartitionTable &partition)
{
  if (partition.groups.size() == 0)
  {
    std::cerr << "There is no 'backends' information";
    return false;
  }
  if (partition.default_group.empty())
  {
    std::cerr << "There is no 'default' backend information";
    return false;
  }
  if (!partee::is_one_of(partition.default_group, partition.groups))
  {
    std::cerr << "'default' backend is not one of 'backends' item";
    return false;
  }
  for (auto &byopcode : partition.byopcodes)
  {
    if (!partee::is_one_of(byopcode.second, partition.groups))
    {
      std::cerr << "OPCODE " << byopcode.first << " is not assigned to one of 'backends' items";
      return false;
    }
  }
  return true;
}

void dump(std::ostream &os, const luci::PartitionTable &table)
{
  os << "Backends:";
  for (auto &group : table.groups)
  {
    os << " " << group;
    if (table.default_group == group)
      os << "(default)";
  }
  os << std::endl;

  os << "Assign by OPCODE: " << std::endl;
  for (auto &item : table.byopcodes)
    os << "  " << item.first << "=" << item.second << std::endl;
}

std::ostream &operator<<(std::ostream &os, const luci::PartitionTable &table)
{
  dump(os, table);
  return os;
}

} // namespace

int entry(int argc, char **argv)
{
  LOGGER(l);

  arser::Arser arser("circle-partitioner provides circle model partitioning");

  build_arser(arser);

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cerr << arser;
    return 255;
  }

  std::string partition_file = arser.get<std::string>(opt_part);
  std::string input_file = arser.get<std::string>(opt_input);
  std::string work_folder = arser.get<std::string>(opt_work);

  std::string partition_path = work_folder + "/" + partition_file;
  std::string input_path = work_folder + "/" + input_file;

  auto module = load_model(input_path);
  if (module.get() == nullptr)
  {
    return EXIT_FAILURE;
  }

  // Read partition information
  INFO(l) << "--- Read PartitionConfig-----------------------" << std::endl;
  auto partition = partee::read(partition_path);
  INFO(l) << partition << std::endl;

  // override with command line arguments
  {
    if (arser[opt_bks])
    {
      auto backend_backends = arser.get<std::string>(opt_bks);
      partition.groups = partee::csv_to_vector<std::string>(backend_backends);
    }
    if (arser[opt_def])
    {
      partition.default_group = arser.get<std::string>(opt_def);
    }
  }
  if (!validate_partition(partition))
  {
    return EXIT_FAILURE;
  }

  INFO(l) << "--- PartitionConfig final----------------------" << std::endl;
  INFO(l) << partition << std::endl;

  // apply partition to module
  auto pms = luci::apply(module.get(), partition);
  for (auto &pmodule : pms.pmodules)
  {
    for (size_t g = 0; g < module->size(); ++g)
    {
      auto graph = module->graph(g);
      if (graph == nullptr)
      {
        std::cerr << "ERROR: Failed to create partition model" << std::endl;
        return EXIT_FAILURE;
      }
      if (!luci::validate(graph))
      {
        std::cerr << "ERROR: Failed to create partition model" << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  INFO(l) << "--- Partition Export---------------------------" << std::endl;
  uint32_t idx = 1;
  for (auto &pmodule : pms.pmodules)
  {
    // Export to output circle file
    luci::CircleExporter exporter;

    auto output_path = partee::make_path(work_folder, input_path, idx, pmodule.group);
    pmodule.name = partee::get_filename_ext(output_path);
    INFO(l) << "--- " << output_path << ": " << pmodule.name << std::endl;

    luci::CircleFileExpContract contract(pmodule.module.get(), output_path);
    if (!exporter.invoke(&contract))
    {
      std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
      return EXIT_FAILURE;
    }
    idx++;
  }

  // TODO add implementation

  return 0;
}
