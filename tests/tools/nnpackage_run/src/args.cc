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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <iostream>

namespace nnpkg_run
{

// param shape_str is a form of, e.g., "[1, [2, 3], 3, []"
void setShape(TensorShapeMap &shape_map, const std::string &shape_str)
{
  using boost::property_tree::ptree;
  std::stringstream ss(shape_str);

  ptree pt;
  read_json(ss, pt);

  int i = 0;
  int tensor_ind = -1;
  for (auto &item : pt.get_child(""))
  {
    if (i % 2 == 0)
    {
      tensor_ind = item.second.get_value<int>();
    }
    else
    {
      auto shape_pt = item.second;
      std::vector<int> shape_v;

      for (auto &dim : shape_pt.get_child(""))
      {
        shape_v.emplace_back(dim.second.get_value<int>());
      }
      shape_map[tensor_ind] = shape_v;
    }
    i++;
  }
}

Args::Args(const int argc, char **argv) noexcept
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  // General options
  po::options_description general("General options", 100);

  // clang-format off
  general.add_options()
    ("help,h", "Print available options")
    ("version", "Print version and exit immediately")
    ("nnpackage", po::value<std::string>()->required())
    ("dump,d", po::value<std::string>()->default_value(""), "Output filename")
    ("load,l", po::value<std::string>()->default_value(""), "Input filename")
    ("num_runs,r", po::value<int>()->default_value(1), "The number of runs")
    ("warmup_runs,w", po::value<int>()->default_value(1), "The number of warmup runs")
    ("gpumem_poll,g", po::value<bool>()->default_value(false), "Check gpu memory polling separately")
    ("mem_poll,m", po::value<bool>()->default_value(false), "Check memory polling")
    ("write_report,p", po::value<bool>()->default_value(false),
         "Write report\n"
         "{exec}-{nnpkg}-{backend}.csv will be generated.\n"
         "e.g. nnpackage_run-UNIT_Add_000-acl_cl.csv.\n"
         "{nnpkg} name may be changed to realpath if you use symbolic-link.")
    ("shape_compile", po::value<std::string>()->default_value("[]"),
         "set shape of specified tensor before compilation\n"
         "e.g. '[0, [1, 2], 2, []]' to set 0th tensor to [1, 2] and 2nd tensor to [].\n")
    ("shape_exec", po::value<std::string>()->default_value("[]"),
         "set shape of specified tensor right before execution\n"
         "e.g. '[1, [1, 2]]` to set 1st tensor to [1, 2].\n")

    ;
  // clang-format on

  _options.add(general);
  _positional.add("nnpackage", 1);
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);

  {
    auto conflicting_options = [&](const std::string &o1, const std::string &o2) {
      if ((vm.count(o1) && !vm[o1].defaulted()) && (vm.count(o2) && !vm[o2].defaulted()))
      {
        throw boost::program_options::error(std::string("Two options '") + o1 + "' and '" + o2 +
                                            "' cannot be given at once.");
      }
    };
  }

  if (vm.count("help"))
  {
    std::cout << "nnpackage_run\n\n";
    std::cout << "Usage: " << argv[0] << " path to nnpackage root directory [<options>]\n\n";
    std::cout << _options;
    std::cout << "\n";

    exit(0);
  }

  if (vm.count("version"))
  {
    _print_version = true;
    return;
  }

  po::notify(vm);

  if (vm.count("dump"))
  {
    _dump_filename = vm["dump"].as<std::string>();
  }

  if (vm.count("load"))
  {
    _load_filename = vm["load"].as<std::string>();
  }

  if (vm.count("nnpackage"))
  {
    _package_filename = vm["nnpackage"].as<std::string>();

    if (_package_filename.empty())
    {
      // TODO Print usage instead of the below message
      std::cerr << "Please specify nnpackage file. Run with `--help` for usage."
                << "\n";

      exit(1);
    }
    else
    {
      if (access(_package_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "nnpackage not found: " << _package_filename << "\n";
      }
    }
  }

  if (vm.count("num_runs"))
  {
    _num_runs = vm["num_runs"].as<int>();
  }

  if (vm.count("warmup_runs"))
  {
    _warmup_runs = vm["warmup_runs"].as<int>();
  }

  if (vm.count("gpumem_poll"))
  {
    _gpumem_poll = vm["gpumem_poll"].as<bool>();
  }

  if (vm.count("mem_poll"))
  {
    _mem_poll = vm["mem_poll"].as<bool>();
  }

  if (vm.count("write_report"))
  {
    _write_report = vm["write_report"].as<bool>();
  }

  if (vm.count("shape_compile"))
  {
    auto shape_str = vm["shape_compile"].as<std::string>();
    try
    {
      setShape(_shape_compile, shape_str);
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--shape_compile' option: " << shape_str << std::endl;
      exit(1);
    }
  }

  if (vm.count("shape_exec"))
  {
    auto shape_str = vm["shape_exec"].as<std::string>();
    try
    {
      setShape(_shape_exec, shape_str);
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--shape_exec' option: " << shape_str << std::endl;
      exit(1);
    }
  }
}

} // end of namespace nnpkg_run
