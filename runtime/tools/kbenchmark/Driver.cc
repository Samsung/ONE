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
#include "ConfigFile.h"
#include "OperationLoader.h"

#include <nonius/nonius.h++>

#include <iostream>
#include <string>

// NOTE dlfcn.h is not a standard library
#include <dlfcn.h>

using namespace kbenchmark;

int main(int argc, char *argv[])
{
  Args args(argc, argv);

  // nonius::benchmark_registry aka std::vector<nonius::benchmark>
  nonius::benchmark_registry benchmarks;

  // Load kernel library
  const std::vector<std::string> &kernel_list = args.kernel();
  std::vector<void *> khandle_list;

  for (auto &k : kernel_list)
  {
    void *khandle;
    typedef nonius::benchmark_registry &(*benchmark_entry)(void);
    benchmark_entry kbenchmark_entry;
    // TODO Check if the paramters are valid

    khandle = dlopen(k.c_str(), RTLD_LAZY);
    if (khandle == nullptr)
    {
      std::cerr << "Fail to dlopen " << k << std::endl;
      return EINVAL;
    }
    char *error;
    kbenchmark_entry = reinterpret_cast<benchmark_entry>(dlsym(khandle, "benchmark_functions"));
    if ((error = dlerror()) != nullptr)
    {
      dlclose(khandle);
      std::cerr << error << std::endl;
      return EINVAL;
    }

    // Save khandle for dlclose
    khandle_list.push_back(khandle);

    // Add current kernel benchmark functions to gloal benchmark list
    nonius::benchmark_registry &kbenchmarks = kbenchmark_entry();
    benchmarks.insert(std::end(benchmarks), std::begin(kbenchmarks), std::end(kbenchmarks));
  }

  // Set default test name
  std::string config_name{args.config()};
  config_name = config_name.substr(config_name.find_last_of("/") + 1);
  config_name = config_name.substr(0, config_name.find_last_of("."));
  std::string test_name{"test_benchmark_" + config_name};
  if (!args.output().empty())
  {
    test_name += (std::string{"_"} + args.output());
  }
  std::cout << "Benchmark test name\n    " << test_name << std::endl;

  if (args.verbose())
  {
    std::cout << "benchmark functions list:" << std::endl;
    for (auto &&f : benchmarks)
    {
      if (std::regex_match(f.name, std::regex(args.filter())))
      {
        std::cout << "    " << f.name << std::endl;
      }
    }
  }

  std::string reporter{args.reporter()};
  std::string ext{"." + reporter};
  if (reporter == "standard")
  {
    ext = ".txt";
  }

  // Set noninus configuration
  nonius::configuration cfg;
  cfg.reporter = reporter;
  cfg.filter_pattern = args.filter();
  cfg.verbose = args.verbose();
  cfg.title = test_name;
  cfg.output_file = test_name + ext;
  cfg.summary = true;

  // Create ConfigFile object from config file
  ConfigFile cf(args.config());

  // Get OperationLoader instance
  OperationLoader &opl = OperationLoader::getInstance();

  if (!opl.is_valid(cf.name()))
  {
    std::cerr << cf.name() << " is not valid operation" << std::endl;
  }
  else
  {
    for (auto &c : cf)
    {
      if (reporter != "html")
      {
        std::string temp_name{test_name + std::string{"_"} + std::to_string(c.first)};
        cfg.title = temp_name;
        cfg.output_file = temp_name + ext;
      }

      nonius::parameters op_params = opl[cf.name()]->params(c.first, c.second);
      cfg.params.map = cfg.params.map.merged(op_params);

      nonius::go(cfg, benchmarks);
    }
  }

  // Release kernel library
  benchmarks.clear();
  for (auto khandle : khandle_list)
  {
    dlclose(khandle);
  }

  return 0;
}
