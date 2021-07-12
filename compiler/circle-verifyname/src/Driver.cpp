/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci/Importer.h>
#include <luci/Service/Validate.h>
#include <foder/FileLoader.h>
#include <arser/arser.h>

#include <iostream>
#include <memory>
#include <string>

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

int entry(int argc, char **argv)
{
  arser::Arser arser;
  arser.add_argument("circle")
    .type(arser::DataType::STR)
    .help("Circle file path to check node name uniqueness");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  int result = EXIT_FAILURE;
  std::string model_file = arser.get<std::string>("circle");

  std::cout << "[ RUN       ] Check " << model_file << std::endl;

  auto module = load_model(model_file);
  if (module.get() != nullptr)
  {
    if (validate(module.get()))
    {
      result = EXIT_SUCCESS;
    }
  }

  if (result == EXIT_SUCCESS)
  {
    std::cout << "[      PASS ] Check " << model_file << std::endl;
  }
  else
  {
    std::cout << "[      FAIL ] Check " << model_file << std::endl;
  }

  return result;
}
