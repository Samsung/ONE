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

#include <functional>
#include <iostream>
#include <json/json.h>
#include <stdexcept>

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

// param vector_map_str is a form of, e.g., "[1, [2, 3], 3, []]" or "h5"
template <typename T>
void handleVectorMapJsonParam(std::unordered_map<uint32_t, std::vector<T>> &vector_map,
                              const std::string &vector_map_str)
{
  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(vector_map_str, root, false))
  {
    std::cerr << "Invalid JSON format \"" << vector_map_str << "\"\n";
    exit(1);
  }

  auto arg_map = argArrayToMap(root);
  for (auto &pair : arg_map)
  {
    uint32_t key = pair.first;
    Json::Value &vec_json = pair.second;
    if (!vec_json.isArray())
    {
      throw std::runtime_error("All the values must be list: " + vector_map_str + "\n");
    }

    std::vector<T> vec;
    for (auto &dim_json : vec_json)
    {
      if (dim_json.isUInt())
      {
        vec.emplace_back(dim_json.asUInt64());
      }
      else if (dim_json.isBool())
      {
        vec.emplace_back(dim_json.asBool());
      }
      else
      {
        throw std::runtime_error("The type of " + dim_json.asString() + "in the list \"" +
                                 vector_map_str + "\" is not supported");
      }
    }
    vector_map[key] = vec;
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
  auto process_nnpackage = [&](const std::string &package_filename) {
    _package_filename = package_filename;

    std::cerr << "Package Filename " << _package_filename << std::endl;
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
  };

  auto process_output_sizes = [&](const std::string &output_sizes_json_str) {
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
  };

  auto process_shape_prepare = [&](const std::string &shape_str) {
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (shape_str == "H5" || shape_str == "h5")
    {
      _when_to_use_h5_shape = WhenToUseH5Shape::PREPARE;
      return;
    }
#endif
    try
    {
      handleVectorMapJsonParam(_shape_prepare, shape_str);
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--shape_prepare' option: " << shape_str << std::endl;
      exit(1);
    }
  };

  auto process_shape_run = [&](const std::string &shape_str) {
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (shape_str == "H5" || shape_str == "h5")
    {
      _when_to_use_h5_shape = WhenToUseH5Shape::RUN;
      return;
    }
#endif
    try
    {
      handleVectorMapJsonParam(_shape_run, shape_str);
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--shape_run' option: " << shape_str << std::endl;
      exit(1);
    }
  };

  auto process_parallel_inputs = [&](const std::string &parallel_inputs_str) {
    try
    {
      handleVectorMapJsonParam(_parallel_inputs, parallel_inputs_str);

      // if (batches.size() != 0)
      // {
      // throw std::runtime_error("Duplicated batches for parallel execution");
      // }
      //
      // if (!reader.parse(parallel_inputs_str, root, false))
      // {
      // throw std::runtime_error("Invalid JSON format for parallel_inputs");
      // }
    }
    catch (const std::exception &e)
    {
      std::cerr << "error with '--parallel_inputs' option: " << parallel_inputs_str << std::endl;
      std::cerr << e.what() << std::endl;
      exit(1);
    }
  };

  // General options
  po::options_description general("General options", 100);

  // clang-format off
  general.add_options()
    ("help,h", "Print available options")
    ("version", "Print version and exit immediately")
    ("nnpackage", po::value<std::string>()->required()->notifier(process_nnpackage))
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    ("dump,d", po::value<std::string>()->default_value("")->notifier([&](const auto &v) { _dump_filename = v; }), "Output filename")
    ("load,l", po::value<std::string>()->default_value("")->notifier([&](const auto &v) { _load_filename = v; }), "Input filename")
#endif
    ("dump:raw", po::value<std::string>()->default_value("")->notifier([&](const auto &v) { _dump_raw_filename = v; }), "Raw Output filename")
    ("load:raw", po::value<std::string>()->default_value("")->notifier([&](const auto &v) { _load_raw_filename = v; }), "Raw Input filename")
    ("output_sizes", po::value<std::string>()->notifier(process_output_sizes),
        "The output buffer size in JSON 1D array\n"
        "If not given, the model's output sizes are used\n"
        "e.g. '[0, 40, 2, 80]' to set 0th tensor to 40 and 2nd tensor to 80.\n")
    ("num_runs,r", po::value<int>()->default_value(1)->notifier([&](const auto &v) { _num_runs = v; }), "The number of runs")
    ("warmup_runs,w", po::value<int>()->default_value(0)->notifier([&](const auto &v) { _warmup_runs = v; }), "The number of warmup runs")
    ("run_delay,t", po::value<int>()->default_value(-1)->notifier([&](const auto &v) { _run_delay = v; }), "Delay time(us) between runs (as default no delay")
    ("gpumem_poll,g", po::value<bool>()->default_value(false)->notifier([&](const auto &v) { _gpumem_poll = v; }), "Check gpu memory polling separately")
    ("mem_poll,m", po::value<bool>()->default_value(false)->notifier([&](const auto &v) { _mem_poll = v; }), "Check memory polling")
    ("write_report,p", po::value<bool>()->default_value(false)->notifier([&](const auto &v) { _write_report = v; }),
         "Write report\n"
         "{exec}-{nnpkg}-{backend}.csv will be generated.\n"
         "e.g. nnpackage_run-UNIT_Add_000-acl_cl.csv.\n"
         "{nnpkg} name may be changed to realpath if you use symbolic-link.")
    ("shape_prepare", po::value<std::string>()->default_value("[]")->notifier(process_shape_prepare),
         "Please refer to the description of 'shape_run'")
    ("shape_run", po::value<std::string>()->default_value("[]")->notifier(process_shape_run),
         "'--shape_prepare: set shape of tensors before compilation (before calling nnfw_prepare()).\n"
         "'--shape_run: set shape of tensors before running (before calling nnfw_run()).\n"
         "Allowed value:.\n"
         "'[0, [1, 2], 2, []]': set 0th tensor to [1, 2] and 2nd tensor to [] (scalar).\n"
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
         "'h5': read shape(s) from H5 input file. '--load' should also be provided.\n"
         "if '--load' option is provided but '--shape_prepare' or '--shape_run' is not provided,\n"
         "'--shape_run h5' will be used by default.\n"
#endif
         "For detailed description, please consutl the description of nnfw_set_input_tensorinfo()\n"
         )
    ("parallel_inputs",po::value<std::string>()->default_value("[]")->notifier(process_parallel_inputs),
     "'--parallel_inputs': Index list of inputs which has batches to be executed in parallel.\n"
     "                     This option also means to execute a model in parallel.\n"
     "                     Currently, only the 'trix' backend supports this option. So, please use this option with the env 'BACKENDs=trix'\n"
     "                     Inputs to be used must have the same batch sizes and ranks.\n"
     "Allowed value:.\n"
         "'[0, [true, false, false, false], 2, [true]]': set 0th tensor and 2nd tensor as parallel inputs.\n")

    ("verbose_level,v", po::value<int>()->default_value(0)->notifier([&](const auto &v) { _verbose_level = v; }),
         "Verbose level\n"
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

    // calling, e.g., "nnpackage_run .. -- shape_prepare .. --shape_run .." should theoretically
    // work but allowing both options together on command line makes the usage and implemenation
    // of nnpackage_run too complicated. Therefore let's not allow those option together.
    conflicting_options("shape_prepare", "shape_run");
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

  try
  {
    po::notify(vm);
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "Bad cast error - " << e.what() << '\n';
    exit(1);
  }

  // This must be run after `notify` as `_warm_up_runs` must have been processed before.
  if (vm.count("mem_poll"))
  {
    // Instead of EXECUTE to avoid overhead, memory polling runs on WARMUP
    if (_mem_poll && _warmup_runs == 0)
    {
      _warmup_runs = 1;
    }
  }

  if (vm.count("parallel_inputs"))
  {
    // Verify _parallel_inputs with _shape_run
    if (_parallel_inputs.size() != 0 && _shape_prepare.size() == 0 && _shape_run.size() == 0)
    {
      std::cerr
        << "Use '--parallel_inputs' option with the option '--shape_prepare' or '--shape_run'"
        << std::endl;
      exit(-1);
    }

    std::unordered_map<uint32_t, uint32_t> parallel_batch_dim_map;
    for (const auto &pair : _parallel_inputs)
    {
      const auto input_index = pair.first;
      const auto &is_batch_vec = pair.second;

      auto batch_count = 0;
      auto batch_dim = 0;
      for (uint32_t dim = 0; dim < is_batch_vec.size(); ++dim)
      {
        if (is_batch_vec[dim])
        {
          batch_count++;
          batch_dim = dim;
        }
      }

      // Check if input has multiple batches
      if (batch_count > 1)
      {
        std::cerr << "Invalid number of batches, '--parallel_inputs' input index '" << input_index
                  << "' has mulitple batches." << std::endl;
        exit(-1);
      }

      // Set batch dimension
      if (batch_count == 1)
      {
        parallel_batch_dim_map[input_index] = batch_dim;
      }
    }

    if (parallel_batch_dim_map.size() > 0)
    {
      const auto check_parallel_inputs_with_resizing_shape_option =
        [&](const nnpkg_run::TensorShapeMap &shape_map) {
          const auto it = parallel_batch_dim_map.begin();
          const auto index = it->first;
          if (shape_map.find(index) == shape_map.end())
          {
            std::cerr << "Unmatched input index, '--parallel_inputs' input index '" << index
                      << "' does not exists in an option resetting shape." << std::endl;
            exit(-1);
          }

          const auto &shape_with_parallel_batch = shape_map.at(index);
          const auto batch_size = shape_with_parallel_batch[parallel_batch_dim_map[index]];
          for (const auto pair : _parallel_inputs)
          {
            const auto input_index = pair.first;
            const auto &is_batch_vec = pair.second;

            // Check if input index exists in parallel_inputs but not in shape_map
            if (shape_map.find(input_index) == shape_map.end())
            {
              std::cerr << "Unmatched input index, '--parallel_inputs' input index '" << input_index
                        << "' does not exists in an option resetting shape." << std::endl;
              exit(-1);
            }

            if (parallel_batch_dim_map.find(input_index) != parallel_batch_dim_map.end())
            {
              // Check if the ranks in _parallel_inputs are not the same as ranks of shape_map
              if (shape_map.at(input_index).size() != is_batch_vec.size())
              {
                std::cerr << "Unmatched ranks, '--parallel_inputs' input index " << input_index
                          << "'s rank is not the same as rank in an option resetting shape."
                          << std::endl;
                std::cerr << "                 '--parallel_inputs' index '" << input_index
                          << "' rank : " << is_batch_vec.size() << std::endl;
                std::cerr << "                 '--shape_prepare'   index '" << input_index
                          << "' rank : " << shape_map.at(input_index).size() << std::endl;
                exit(-1);
              }

              // Check if the batch size is less than 2
              const auto batch_dim = parallel_batch_dim_map[input_index];
              if (shape_map.at(input_index).at(batch_dim) < 2)
              {
                std::cerr << "Invalid batch size for using '--parallel_inputs' option : input '"
                          << input_index << "'s batch size is " << batch_size << std::endl;
                exit(-1);
              }

              // Check if all batches are equal
              if (batch_size != shape_map.at(input_index).at(batch_dim))
              {
                std::cerr << "Unmatched batch sizes for using '--parallel_inputs' option'"
                          << std::endl;
                exit(-1);
              }
            }
          }
        };

      if (_shape_prepare.size() != 0)
      {
        check_parallel_inputs_with_resizing_shape_option(_shape_prepare);
      }

      if (_shape_run.size() != 0)
      {
        check_parallel_inputs_with_resizing_shape_option(_shape_run);
      }
    }
  }
}

bool Args::shapeParamProvided()
{
  bool provided = false;
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
  // "--shape_run h5" or "--shape_prepare h5" was provided
  provided = (getWhenToUseH5Shape() != WhenToUseH5Shape::NOT_PROVIDED);
#endif
  // specific shape was provided
  // e.g., "--shape_run '[0, [10, 1]]'" or "--shape_prepare '[0, [10, 1]]'"
  provided |= (!getShapeMapForPrepare().empty()) || (!getShapeMapForRun().empty());

  return provided;
}

} // end of namespace nnpkg_run
