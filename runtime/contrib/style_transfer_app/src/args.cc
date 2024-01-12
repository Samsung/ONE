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
  po::options_description general("General options");

  // clang-format off
  general.add_options()
    ("help,h", "Display available options")
    ("nnpackage", po::value<std::string>()->required(), "nnpackage path")
    ("input,i", po::value<std::string>()->required(), "Input image path")
    ("output,o", po::value<std::string>()->required(), "Output image path");
  // clang-format on

  _options.add(general);
  _positional.add("nnpackage", 1);
}

void Args::Parse(const int argc, char **argv)
{

  po::variables_map vm;
  try
  {
    po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
              vm);

    if (vm.count("help"))
    {
      std::cout << "style_transfer_app\n\n";
      std::cout << "Usage: " << argv[0] << " path to nnpackage root directory [<options>]\n\n";
      std::cout << _options;
      std::cout << "\n";

      exit(0);
    }

    po::notify(vm);

    if (vm.count("input"))
    {
      _input_filename = vm["input"].as<std::string>();
    }

    if (vm.count("output"))
    {
      _output_filename = vm["output"].as<std::string>();
    }

    if (vm.count("nnpackage"))
    {
      _package_filename = vm["nnpackage"].as<std::string>();

      if (!std::filesystem::exists(_package_filename))
      {
        std::cerr << "nnpackage not found: " << _package_filename << "\n";
      }
    }
  }
  catch (const boost::program_options::required_option &e)
  {
    std::cerr << e.what() << std::endl;
    return exit(-1);
  }
}

} // namespace StyleTransferApp
