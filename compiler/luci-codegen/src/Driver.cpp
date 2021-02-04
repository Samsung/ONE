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

#include "Codegen.h"
#include "luci/Importer.h"
#include "luci/CircleExporter.h"
#include "luci/CircleFileExpContract.h"

#include <iostream>
#include <string>

std::vector<char> read_file(const std::string &path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    std::cerr << "failed to open \"" << path << "\" file\n";
    exit(1);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (file.read(buffer.data(), size).bad())
  {
    std::cerr << "failed to read \"" << path << "\" file\n";
    exit(1);
  }
  return buffer;
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cout << "Usage: ./circle_codegen <input circle file> <output package name>\n";
    return 1;
  }
  std::string input_circle_name = argv[1];
  std::string output_package_name = argv[2];
  luci::Importer importer;

  auto raw_model_data = read_file(input_circle_name);

  const circle::Model *circle_module = circle::GetModel(raw_model_data.data());
  std::unique_ptr<luci::Module> luci_module = importer.importModule(circle_module);

  luci_codegen::Options options;
  options.generate_checks = false;
  // set options if needed
  luci_codegen::Codegen codegen(options);
  codegen.process_module(*luci_module);
  codegen.emit_code(output_package_name);

  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(luci_module.get(), output_package_name + ".circle");
  exporter.invoke(&contract);
  return 0;
}
