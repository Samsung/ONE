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

#include "EqualizePatternFinder.h"
#include "EqualizePatternWrite.h"

#include <arser/arser.h>
#include <foder/FileLoader.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/Importer.h>
#include <luci/ImporterEx.h>
#include <luci/Service/Validate.h>

#include <iostream>
#include <string>

using namespace fme_detect;

namespace
{

// Return luci::Module from model_path
std::unique_ptr<luci::Module> import(const std::string &model_path)
{
  // Load model from the file
  foder::FileLoader loader{model_path};
  std::vector<char> model_data = loader.load();

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (not circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Failed to verify circle '" + model_path + "'");
  }

  auto circle_model = circle::GetModel(model_data.data());

  if (not circle_model)
    throw std::runtime_error("Failed to load '" + model_path + "'");

  auto module = luci::Importer().importModule(circle_model);

  if (not module)
    throw std::runtime_error("Failed to load '" + model_path + "'");

  return module;
}

} // namespace

int entry(int argc, char **argv)
{
  arser::Arser arser("fme-detect detects patterns to apply FM equalization in "
                     "the circle model");

  arser::Helper::add_verbose(arser);

  arser.add_argument("--input").required().help("Input circle model");

  arser.add_argument("--output")
    .required()
    .help("Output json file that describes equalization patterns");

  arser.add_argument("--allow_dup_op")
    .nargs(0)
    .default_value(false)
    .help("Allow to create duplicate operations when a feature map matches "
          "with multiple equalization patterns. This can increase the size of "
          "the model. Default is false.");

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
  const std::string output_path = arser.get<std::string>("--output");
  const bool allow_dup_op = arser.get<bool>("--allow_dup_op");

  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = allow_dup_op;
  }
  EqualizePatternFinder finder(ctx);

  auto module = import(input_path);
  assert(module != nullptr); // FIX_ME_UNLESS

  std::vector<EqualizePattern> patterns;
  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    auto matched = finder.find(graph);
    patterns.insert(patterns.end(), matched.begin(), matched.end());
  }

  fme_detect::write(patterns, output_path);

  return EXIT_SUCCESS;
}
