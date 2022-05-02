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
#include "PartitionExport.h"
#include "HelperPath.h"

#include <luci/ImporterEx.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/CircleOptimizer.h>
#include <luci/PartitionDump.h>
#include <luci/PartitionValidate.h>
#include <luci/Log.h>

#include <pepper/csv2vec.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>

namespace
{

const char *opt_bks = "--backends";
const char *opt_def = "--default";
const char *opt_part_file = "--part_file";
const char *opt_input_file = "--input_file";
const char *opt_work_path = "--work_path";

void print_version(void)
{
  std::cout << "circle-partitioner version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

void build_arser(arser::Arser &arser)
{
  arser::Helper::add_version(arser, print_version);

  arser.add_argument(opt_bks).help("Backends in CSV to use for partitioning");

  arser.add_argument(opt_def).help("Default backend to assign");

  arser.add_argument(opt_part_file)
    .required(true)
    .help("Partition file which provides backend to assign");
  arser.add_argument(opt_input_file).required(true).help("Input circle model filename");
  arser.add_argument(opt_work_path)
    .help("Work folder of partition, input files exist and output files are produced");
}

std::unique_ptr<luci::Module> load_model(const std::string &input_path)
{
  // Import from input Circle file
  luci::ImporterEx importerex;
  return importerex.importVerifyModule(input_path);
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
    return EXIT_FAILURE;
  }

  std::string partition_file = arser.get<std::string>(opt_part_file);
  std::string input_file = arser.get<std::string>(opt_input_file);
  std::string work_folder = ".";

  if (arser[opt_work_path])
  {
    work_folder = arser.get<std::string>(opt_work_path);
  }

  std::string partition_path = work_folder + "/" + partition_file;
  std::string input_path = work_folder + "/" + input_file;

  auto module = load_model(input_path);
  if (module.get() == nullptr)
  {
    return EXIT_FAILURE;
  }
  // Run default shape/dtype inference before validation
  // NOTE CircleWhileOut default shape is INVALID as it needs initial shape
  //      inference. This is cause of WHILE may have dynamic shape.
  luci::CircleOptimizer optimizer;
  (void)optimizer.options(); // need to call this to make internal member
  for (size_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);
    optimizer.optimize(graph);
  }
  if (!luci::validate(module.get()))
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
      partition.groups = pepper::csv_to_vector<std::string>(backend_backends);
    }
    if (arser[opt_def])
    {
      partition.default_group = arser.get<std::string>(opt_def);
    }
  }
  if (!luci::validate(partition))
  {
    // NOTE error reason/message is put to std::cerr inside validate()
    return EXIT_FAILURE;
  }

  INFO(l) << "--- PartitionConfig final----------------------" << std::endl;
  INFO(l) << partition << std::endl;

  // apply partition to module
  auto pms = luci::apply(module.get(), partition);

  // validate partitioned modules
  for (auto &pmodule : pms.pmodules)
  {
    for (size_t g = 0; g < pmodule.module->size(); ++g)
    {
      auto graph = pmodule.module->graph(g);
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

  INFO(l) << "--- Partition connection information-----------" << std::endl;
  if (!partee::export_part_conn_json(work_folder, input_file, module.get(), pms))
  {
    return EXIT_FAILURE;
  }
  if (!partee::export_part_conn_ini(work_folder, input_file, module.get(), pms))
  {
    return EXIT_FAILURE;
  }

  INFO(l) << "--- Partition done-----------------------------" << std::endl << std::endl;

  return EXIT_SUCCESS;
}
