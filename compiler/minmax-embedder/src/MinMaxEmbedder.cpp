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

#include <H5Cpp.h>

#include <luci/ImporterEx.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>

namespace
{

void print_version(void)
{
  std::cout << "minmax-embedder version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

} // namespace

int entry(int argc, char **argv)
{
  arser::Arser arser("minmax-embedder embeds given minmax into circle");
  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);
  // named args
  arser.add_argument("--min_percentile")
    .type(arser::DataType::FLOAT)
    .default_value(1.f)
    .help("Set min percentile");
  arser.add_argument("--max_percentile")
    .type(arser::DataType::FLOAT)
    .default_value(99.f)
    .help("Set max percentile");
  arser.add_argument("-o").default_value("out.circle").help("Path to output circle model");
  // positional args: minmax(h5), input(circle)
  arser.add_argument("circle").help("Path to input circle model");
  arser.add_argument("minmax").help("Path to minmax data in hdf5");
  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  std::string mm_path = arser.get<std::string>("minmax");
  std::string ic_path = arser.get<std::string>("circle");
  std::string oc_path = arser.get<std::string>("-o");

  H5::H5File _file;
  H5::Group _group;

  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(ic_path);
  if (module.get() == nullptr)
    return EXIT_FAILURE;

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    // embed minmax
    // embed(graph);

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return 255;
    }
  }

  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(module.get(), oc_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << oc_path << "'" << std::endl;
    return 255;
  }

  return 0;
}
