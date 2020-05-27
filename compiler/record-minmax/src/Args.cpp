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

#include "Args.h"

#include <iostream>

namespace record_minmax
{

Args::Args(const int argc, char **argv)
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  po::options_description desc("Allowed options");

  desc.add_options()("help,h", "Print available options")(
      "input_model,i", po::value<std::string>()->default_value(""), "Input model filename")(
      "input_data,d", po::value<std::string>()->default_value(""), "Input data filename")(
      "output_model,o", po::value<std::string>()->default_value(""), "Output model filename");

  _positional.add("input_model", 1).add("input_data", 1).add("output_model", 1);
  _options.add(desc);
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << "record-minmax\n" << std::endl;
    std::cout << "Usage: " << argv[0]
              << " <path/to/input/model> <path/to/input/data> <path/to/output/model>" << std::endl;
    std::cout << _options << std::endl;

    exit(EXIT_SUCCESS);
  }

  if (vm.count("input_model"))
  {
    _input_model_filename = vm["input_model"].as<std::string>();

    if (_input_model_filename.empty())
    {
      std::cerr << "Please specify input model file. Run with `--help` for usage." << std::endl;
      exit(EXIT_FAILURE);
    }
    else
    {
      if (access(_input_model_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "input model file not found: " << _input_model_filename << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (vm.count("input_data"))
  {
    _input_data_filename = vm["input_data"].as<std::string>();

    if (_input_data_filename.empty())
    {
      std::cerr << "Please specify input data file. Run with `--help` for usage." << std::endl;
      exit(EXIT_FAILURE);
    }
    else
    {
      if (access(_input_data_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "input data file not found: " << _input_data_filename << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (vm.count("output_model"))
  {
    _output_model_filename = vm["output_model"].as<std::string>();

    if (_output_model_filename.empty())
    {
      std::cerr << "Please specify output model file. Run with `--help` for usage." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

} // end of namespace record_minmax
