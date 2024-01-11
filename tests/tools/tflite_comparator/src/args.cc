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

namespace TFLiteRun
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
    ("tflite", po::value<std::string>()->default_value("")->required(), "Input tflite model file for serialization")
    ("data,d", po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{}, ""), "Input data file for model");
  // clang-format on

  _options.add(general);
  _positional.add("tflite", 1);
}

void Args::print(char **argv)
{
  std::cout << "tflite_comparator" << std::endl << std::endl;
  std::cout << "Load tflite model by onert and TFLite, and compare their output" << std::endl;
  std::cout << "Usage:" << std::endl;
  std::cout << argv[0] << " --tflite model_file.tflite --data input_data.dat" << std::endl;
  std::cout << _options;
  std::cout << std::endl;
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    print(argv);

    exit(0);
  }

  try
  {
    if (vm.count("tflite"))
    {
      _tflite_filename = vm["tflite"].as<std::string>();
    }

    if (vm.count("data"))
    {
      _data_filenames = vm["data"].as<std::vector<std::string>>();
    }
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << e.what() << '\n';
    print(argv);
    exit(1);
  }
}

} // end of namespace TFLiteRun
