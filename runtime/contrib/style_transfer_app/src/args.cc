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

#include "args.h"

#include <iostream>
#include <filesystem>

namespace StyleTransferApp
{

Args::Args(const int argc, char **argv) noexcept
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  // General options
  _arser.add_argument("nnpackage").type(arser::DataType::STR).help("nnpackage path");
  _arser.add_argument("--input", "-i")
    .type(arser::DataType::STR)
    .required()
    .help("Input image path");
  _arser.add_argument("--output", "-o")
    .type(arser::DataType::STR)
    .required()
    .help("Output image path");
}

void Args::Parse(const int argc, char **argv)
{
  try
  {
    _arser.parse(argc, argv);

    _input_filename = _arser.get<std::string>("--input");
    _output_filename = _arser.get<std::string>("--output");

    _package_filename = _arser.get<std::string>("nnpackage");
    if (!std::filesystem::exists(_package_filename))
    {
      std::cerr << "nnpackage not found: " << _package_filename << "\n";
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return exit(-1);
  }
}

} // namespace StyleTransferApp
