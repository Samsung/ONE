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

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>

namespace
{

const char *opt_prt = "--partition";
const char *opt_bks = "--backends";
const char *opt_def = "--default";

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

  arser.add_argument(opt_prt)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Partition information file which provides backend to assign");

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

  arser.add_argument("input")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Input circle model file path");
  arser.add_argument("output")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Output parition folder path");
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

} // namespace

int entry(int argc, char **argv)
{
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

  std::string input_path = arser.get<std::string>("input");

  auto module = load_model(input_path);
  if (module.get() == nullptr)
  {
    return EXIT_FAILURE;
  }

  // TODO add implementation
  (void)module;

  return 0;
}
