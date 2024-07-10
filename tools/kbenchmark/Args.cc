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

#include <arser/arser.h>

#include <iostream>
#include <filesystem>

namespace kbenchmark
{

Args::Args(const int argc, char **argv) noexcept { Initialize(argc, argv); }

void Args::Initialize(const int argc, char **argv)
{
  arser::Arser arser;

  arser.add_argument("--config", "-c")
    .type(arser::DataType::STR)
    .required()
    .help("Configuration filename");
  arser.add_argument("--kernel", "-k")
    .type(arser::DataType::STR)
    .accumulated()
    .help("Kernel library name, support multiple kernel libraries");
  arser.add_argument("--reporter", "-r")
    .type(arser::DataType::STR)
    .default_value("standard")
    .help("Set reporter types(standard, html, junit, csv)");
  arser.add_argument("--filter", "-f")
    .type(arser::DataType::STR)
    .default_value(".*")
    .help("Only run benchmarks whose name matches the regular expression pattern");
  arser.add_argument("--verbose", "-v")
    .type(arser::DataType::INT32)
    .default_value(0)
    .help("Show verbose output");
  arser.add_argument("--output", "-o")
    .type(arser::DataType::STR)
    .default_value("")
    .help("Set additional strings for output file name");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    exit(0);
  }

  _config = arser.get<std::string>("--config");
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

  _kernel = arser.get<std::vector<std::string>>("--kernel");
  if (_kernel.size() > 0)
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

  _reporter = arser.get<std::string>("--reporter");
  if (!_reporter.empty())
  {
    if (_reporter != "junit" && _reporter != "csv" && _reporter != "html" &&
        _reporter != "standard")
    {
      std::cerr << "Invalid reporter" << std::endl;
      exit(1);
    }
  }

  _filter = arser.get<std::string>("--filter");
  _output = arser.get<std::string>("--output");
  _verbose = arser.get<int>("--verbose");
}

} // namespace kbenchmark
