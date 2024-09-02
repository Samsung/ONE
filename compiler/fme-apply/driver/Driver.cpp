/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FMEqualizer.h"
#include "EqualizePatternRead.h"

#include <arser/arser.h>
#include <foder/FileLoader.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/ImporterEx.h>
#include <luci/Service/Validate.h>

#include <iostream>
#include <string>

using namespace fme_apply;

int entry(int argc, char **argv)
{
  arser::Arser arser("fm-apply applies channel-wise scale/shift for FM equalization");

  arser::Helper::add_verbose(arser);

  arser.add_argument("--input").required().help("Input circle model");

  arser.add_argument("--fme_patterns")
    .required()
    .help("Json file that describes feture map equalization info");

  arser.add_argument("--output").required().help("Output circle model");

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

  const std::string input_path = arser.get<std::string>("--input");
  const std::string fme_patterns_path = arser.get<std::string>("--fme_patterns");
  const std::string output_path = arser.get<std::string>("--output");

  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(input_path);
  assert(module != nullptr); // FIX_ME_UNLESS

  auto patterns = fme_apply::read(fme_patterns_path);

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    FMEqualizer equalizer;
    equalizer.equalize(graph, patterns);

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Equalized graph is invalid" << std::endl;
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

  return EXIT_SUCCESS;
}
