/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

namespace nnapi_test
{

Args::Args(const int argc, char **argv)
{
  Initialize();
  try
  {
    Parse(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << "The argments that cannot be parsed: " << e.what() << '\n';
    print(argv);
    exit(255);
  }
}

void Args::print(char **argv)
{
  std::cout << "nnapi_test\n\n";
  std::cout << "Usage: " << argv[0] << " <.tflite> [<options>]\n\n";
  std::cout << _options;
  std::cout << "\n";
}

void Args::Initialize(void)
{
  // General options
  po::options_description general("General options", 100);

  // clang-format off
  general.add_options()
    ("help,h", "Print available options")
    ("tflite", po::value<std::string>()->required())
    ("seed", po::value<int>()->default_value(0), "The seed of random inputs")
    ("num_runs", po::value<int>()->default_value(2), "The number of runs")
    ;
  // clang-format on

  _options.add(general);
  _positional.add("tflite", 1);
  _positional.add("seed", 2);
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);

  if (vm.count("help"))
  {
    print(argv);

    exit(0);
  }

  po::notify(vm);
  if (vm.count("tflite"))
  {
    _tflite_filename = vm["tflite"].as<std::string>();

    if (_tflite_filename.empty())
    {
      std::cerr << "Please specify tflite file.\n";
      print(argv);
      exit(255);
    }
    else
    {
      if (access(_tflite_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "tflite file not found: " << _tflite_filename << "\n";
        exit(255);
      }
    }
  }

  if (vm.count("seed"))
  {
    _seed = vm["seed"].as<int>();
  }

  if (vm.count("num_runs"))
  {
    _num_runs = vm["num_runs"].as<int>();
    if (_num_runs < 0)
    {
      std::cerr << "num_runs value must be greater than 0.\n";
      exit(255);
    }
  }
}

} // end of namespace nnapi_test
