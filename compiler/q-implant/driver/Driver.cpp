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

#include <luci/ImporterEx.h>
#include <luci/CircleQuantizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>

#include <arser/arser.h>

#include "QImplant.h"

#include <iostream>
#include <string>

using namespace q_implant;

int entry(int argc, char **argv)
{
  arser::Arser arser("q-implant provides circle model quantization");

  arser::Helper::add_verbose(arser);

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("qparam").help("Quantization parameter file (.json)");
  arser.add_argument("output").help("Output circle model");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  const std::string input_path = arser.get<std::string>("input");
  const std::string qparam_path = arser.get<std::string>("qparam");
  const std::string output_path = arser.get<std::string>("output");

  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(input_path);
  if (module.get() == nullptr)
    return EXIT_FAILURE;

  QImplant writer(qparam_path);

  if (module->size() != 1)
  {
    std::cerr << "ERROR: Only a single subgraph is supported" << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    writer.write(graph);

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(module.get(), output_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
