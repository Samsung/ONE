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

#include "Args.h"

#include <iostream>
#include <filesystem>

namespace kbenchmark
{

Args::Args(const int argc, char **argv) noexcept { Initialize(argc, argv); }

void Args::Initialize(const int argc, char **argv)
{
  // General options
  po::options_description general("General options");
  // clang-format off
  general.add_options()("help,h", "Display available options")
    ("config,c", po::value<std::string>(&_config)->required(), "Configuration filename")
    ("kernel,k", po::value<std::vector<std::string>>(&_kernel)->multitoken()->composing()->required(), "Kernel library name, support multiple kernel libraries")
    ("reporter,r", po::value<std::string>(&_reporter)->default_value("standard"), "Set reporter types(standard, html, junit, csv)")
    ("filter,f", po::value<std::string>(&_filter)->default_value(".*"), "Only run benchmarks whose name matches the regular expression pattern")
    ("verbose,v", po::value<int>(&_verbose)->default_value(0)->implicit_value(true), "Show verbose output")
    ("output,o", po::value<std::string>(&_output)->default_value(""), "Set additional strings for output file name")
  ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, general), vm);

  try
  {
    po::notify(vm);
  }
  catch (const boost::program_options::required_option &e)
  {
    if (vm.count("help"))
    {
      std::cout << general << std::endl;
      exit(0);
    }
    else
    {
      throw e;
    }
  }

  if (vm.count("help"))
  {
    std::cout << general << std::endl;
    exit(0);
  }

  if (vm.count("config"))
  {
    if (_config.substr(_config.find_last_of(".") + 1) != "config")
    {
      std::cerr << "Please specify .config file" << std::endl;
      exit(1);
    }

    if (!std::filesystem::exists(_config))
    {
      std::cerr << _config << " file not found" << std::endl;
      exit(1);
    }
  }

  if (vm.count("kernel"))
  {
    for (auto &k : _kernel)
    {
      if (!std::filesystem::exists(k))
      {
        std::cerr << k << " file not found" << std::endl;
        exit(1);
      }
    }
  }

  if (vm.count("reporter"))
  {
    if (_reporter != "junit" && _reporter != "csv" && _reporter != "html" &&
        _reporter != "standard")
    {
      std::cerr << "Invalid reporter" << std::endl;
      exit(1);
    }
  }
}

} // namespace kbenchmark
