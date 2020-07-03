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
#include <json/json.h>

namespace
{

// This function parses a json object and returns as a vector of integers
// For example,
// [0, [1, 2, 3, 4], 3, 40, 4, []] in JSON
// is converted to:
// {
//  0 -> [1, 2, 3, 4]
//  3 -> 40
//  4 -> []
// } in std::unordered_map. Note that the value type is still Json::Value.
std::unordered_map<uint32_t, Json::Value> argArrayToMap(const Json::Value &jsonval)
{
  if (!jsonval.isArray() || (jsonval.size() % 2 != 0))
  {
    std::cerr << "JSON argument must be an even-sized array in JSON\n";
    exit(1);
  }

  std::unordered_map<uint32_t, Json::Value> ret;
  for (uint32_t i = 0; i < jsonval.size(); i += 2)
  {
    if (!jsonval[i].isUInt())
    {
      std::cerr << "Key values(values in even indices) must be unsigned integers\n";
      exit(1);
    }
    uint32_t key = jsonval[i].asUInt();
    Json::Value val = jsonval[i + 1];
    ret[key] = jsonval[i + 1];
  }
  return ret;
}

// param shape_str is a form of, e.g., "[1, [2, 3], 3, []]"
void handleShapeParam(nnpkg_run::TensorShapeMap &shape_map, const std::string &shape_str)
{
  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(shape_str, root, false))
  {
    std::cerr << "Invalid JSON format for output_sizes \"" << shape_str << "\"\n";
    exit(1);
  }

  auto arg_map = argArrayToMap(root);
  for (auto &pair : arg_map)
  {
    uint32_t key = pair.first;
    Json::Value &shape_json = pair.second;
    if (!shape_json.isArray())
    {
      std::cerr << "All the values must be list: " << shape_str << "\n";
      exit(1);
    }

    std::vector<int> shape;
    for (auto &dim_json : shape_json)
    {
      if (!dim_json.isUInt())
      {
        std::cerr << "All the dims should be dim >= 0: " << shape_str << "\n";
        exit(1);
      }

      shape.emplace_back(dim_json.asUInt64());
    }

    shape_map[key] = shape;
  }
}

} // namespace

namespace nnpkg_run
{

Args::Args(const int argc, char **argv)
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
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    ("dump,d", po::value<std::string>()->default_value(""), "Output filename")
    ("load,l", po::value<std::string>()->default_value(""), "Input filename")
#endif
    ("output_sizes", po::value<std::string>(),
        "The output buffer size in JSON 1D array\n"
        "If not given, the model's output sizes are used\n"
        "e.g. '[0, 40, 2, 80]' to set 0th tensor to 40 and 2nd tensor to 80.\n")
    ("num_runs,r", po::value<int>()->default_value(1), "The number of runs")
    ("warmup_runs,w", po::value<int>()->default_value(0), "The number of warmup runs")
    ("run_delay,t", po::value<int>()->default_value(-1), "Delay time(ms) between runs (as default no delay")
    ("gpumem_poll,g", po::value<bool>()->default_value(false), "Check gpu memory polling separately")
    ("mem_poll,m", po::value<bool>()->default_value(false), "Check memory polling")
    ("write_report,p", po::value<bool>()->default_value(false),
         "Write report\n"
         "{exec}-{nnpkg}-{backend}.csv will be generated.\n"
         "e.g. nnpackage_run-UNIT_Add_000-acl_cl.csv.\n"
         "{nnpkg} name may be changed to realpath if you use symbolic-link.")
    ("shape_prepare", po::value<std::string>()->default_value("[]"),
         "set shape of specified tensor before compilation\n"
         "e.g. '[0, [1, 2], 2, []]' to set 0th tensor to [1, 2] and 2nd tensor to [].\n")
    ("shape_run", po::value<std::string>()->default_value("[]"),
         "set shape of specified tensor right before running\n"
         "e.g. '[1, [1, 2]]` to set 1st tensor to [1, 2].\n")
    ("verbose_level,v", po::value<int>()->default_value(0), "Verbose level\n"
         "0: prints the only result. Messages btw run don't print\n"
         "1: prints result and message btw run\n"
         "2: prints all of messages to print\n")
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
  try
  {
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (vm.count("dump"))
    {
      _dump_filename = vm["dump"].as<std::string>();
    }

    if (vm.count("load"))
    {
      _load_filename = vm["load"].as<std::string>();
    }
#endif

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

    if (vm.count("output_sizes"))
    {
      auto output_sizes_json_str = vm["output_sizes"].as<std::string>();

      Json::Value root;
      Json::Reader reader;
      if (!reader.parse(output_sizes_json_str, root, false))
      {
        std::cerr << "Invalid JSON format for output_sizes \"" << output_sizes_json_str << "\"\n";
        exit(1);
      }

      auto arg_map = argArrayToMap(root);
      for (auto &pair : arg_map)
      {
        uint32_t key = pair.first;
        Json::Value &val_json = pair.second;
        if (!val_json.isUInt())
        {
          std::cerr << "All the values in `output_sizes` must be unsigned integers\n";
          exit(1);
        }
        uint32_t val = val_json.asUInt();
        _output_sizes[key] = val;
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

    if (vm.count("run_delay"))
    {
      _run_delay = vm["run_delay"].as<int>();
    }

    if (vm.count("gpumem_poll"))
    {
      _gpumem_poll = vm["gpumem_poll"].as<bool>();
    }

    if (vm.count("mem_poll"))
    {
      _mem_poll = vm["mem_poll"].as<bool>();
      // Instead of EXECUTE to avoid overhead, memory polling runs on WARMUP
      if (_mem_poll && _warmup_runs == 0)
      {
        _warmup_runs = 1;
      }
    }

    if (vm.count("write_report"))
    {
      _write_report = vm["write_report"].as<bool>();
    }

    if (vm.count("verbose_level"))
    {
      _verbsoe_level = vm["verbose_level"].as<int>();
    }
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "error by bad cast" << e.what() << '\n';
    exit(1);
  }

  if (vm.count("shape_prepare"))
  {
    std::string shape_str;
    try
    {
      shape_str = vm["shape_prepare"].as<std::string>();
    }
    catch (const std::bad_cast &e)
    {
      std::cerr << "error by bad cast with '--shape_prepare' option" << e.what() << '\n';
      exit(1);
    }
    try
    {
      handleShapeParam(_shape_prepare, shape_str);
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--shape_prepare' option: " << shape_str << std::endl;
      exit(1);
    }
  }

  if (vm.count("shape_run"))
  {
    std::string shape_str;
    try
    {
      shape_str = vm["shape_run"].as<std::string>();
    }
    catch (const std::bad_cast &e)
    {
      std::cerr << "error by bad cast with '--shape_run' option" << e.what() << '\n';
      exit(1);
    }
    try
    {
      handleShapeParam(_shape_run, shape_str);
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--shape_run' option: " << shape_str << std::endl;
      exit(1);
    }
  }
}

} // end of namespace nnpkg_run
