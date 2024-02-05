/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "nnfw_util.h"
#include "misc/to_underlying.h"

#include <functional>
#include <iostream>
#include <sys/stat.h>
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

void checkModelfile(const std::string &model_filename)
{
  if (model_filename.empty())
  {
    // TODO Print usage instead of the below message
    std::cerr << "Please specify model file. Run with `--help` for usage."
              << "\n";

    exit(1);
  }
  else
  {
    if (access(model_filename.c_str(), F_OK) == -1)
    {
      std::cerr << "Model file not found: " << model_filename << "\n";
      exit(1);
    }
  }
}

void checkPackage(const std::string &package_filename)
{
  if (package_filename.empty())
  {
    // TODO Print usage instead of the below message
    std::cerr << "Please specify nnpackage file. Run with `--help` for usage."
              << "\n";

    exit(1);
  }
  else
  {
    if (access(package_filename.c_str(), F_OK) == -1)
    {
      std::cerr << "nnpackage not found: " << package_filename << "\n";
      exit(1);
    }
  }
}

// check the value is in the valid_args list and return the corresponded enum
template <typename T>
T checkValidation(const std::string &arg_name, const std::vector<T> &valid_args, int value)
{
  for (const auto arg : valid_args)
  {
    if (value == nnfw::misc::to_underlying(arg))
      return arg;
  }
  std::cerr << arg_name + " " + std::to_string(value) + " is unsupported argument\n";
  exit(1);
}

// generate a help message based on the valid_args and default_arg
template <typename T>
std::string genHelpMsg(const std::string &arg_name, const std::vector<T> &valid_args)
{
  std::string msg = arg_name + "\n";
  msg += "If not given, model's hyper parameter is used\n";
  for (const auto arg : valid_args)
  {
    const auto num = nnfw::misc::to_underlying(arg);
    msg += std::to_string(num) + ": " + onert_train::to_string(arg) + "\n";
  }
  msg.erase(msg.length() - 1); // remove last \n
  return msg;
}

} // namespace

namespace onert_train
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
    checkPackage(package_filename);
  };

  auto process_modelfile = [&](const std::string &model_filename) {
    _model_filename = model_filename;

    std::cerr << "Model Filename " << _model_filename << std::endl;
    checkModelfile(model_filename);

    _use_single_model = true;
  };

  auto process_path = [&](const std::string &path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0)
    {
      if (sb.st_mode & S_IFDIR)
      {
        _package_filename = path;
        checkPackage(path);
        std::cerr << "Package Filename " << path << std::endl;
      }
      else
      {
        _model_filename = path;
        checkModelfile(path);
        std::cerr << "Model Filename " << path << std::endl;
        _use_single_model = true;
      }
    }
    else
    {
      std::cerr << "Cannot find: " << path << "\n";
      exit(1);
    }
  };

  auto process_export_path = [&](const std::string &path) { _export_model_filename = path; };

  auto process_load_raw_inputfile = [&](const std::string &input_filename) {
    _load_raw_input_filename = input_filename;

    std::cerr << "Model Input Filename " << _load_raw_input_filename << std::endl;
    checkModelfile(_load_raw_input_filename);
  };

  auto process_load_raw_expectedfile = [&](const std::string &expected_filename) {
    _load_raw_expected_filename = expected_filename;

    std::cerr << "Model Expected Filename " << _load_raw_expected_filename << std::endl;
    checkModelfile(_load_raw_expected_filename);
  };

  auto process_validation_split = [&](const float v) {
    if (v < 0.f || v > 1.f)
    {
      std::cerr << "Invalid validation_split. Float between 0 and 1." << std::endl;
      exit(1);
    }
    _validation_split = v;
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

  // General options
  po::options_description general("General options", 100);

  // clang-format off
  general.add_options()
    ("help,h", "Print available options")
    ("version", "Print version and exit immediately")
    ("nnpackage", po::value<std::string>()->notifier(process_nnpackage), "NN Package file(directory) name")
    ("modelfile", po::value<std::string>()->notifier(process_modelfile), "NN Model filename")
    ("path", po::value<std::string>()->notifier(process_path), "NN Package or NN Modelfile path")
    ("export_path", po::value<std::string>()->notifier(process_export_path), "Path to export circle")
    ("load_input:raw", po::value<std::string>()->notifier(process_load_raw_inputfile),
         "NN Model Raw Input data file\n"
         "The datafile must have data for each input number.\n"
         "If there are 3 inputs, the data of input0 must exist as much as data_length, "
         "and the data for input1 and input2 must be held sequentially as data_length."
    )
    ("load_expected:raw", po::value<std::string>()->notifier(process_load_raw_expectedfile),
         "NN Model Raw Expected data file\n"
         "(Same data policy with load_input:raw)"
    )
    ("mem_poll,m", po::value<bool>()->default_value(false)->notifier([&](const auto &v) { _mem_poll = v; }), "Check memory polling (default: false)")
    ("epoch", po::value<int>()->default_value(5)->notifier([&](const auto &v) { _epoch = v; }), "Epoch number (default: 5)")
    ("batch_size", po::value<int>()->notifier([&](const auto &v) { _batch_size = v; }), 
      "Batch size\n"
      "If not given, model's hyper parameter is used")
    ("learning_rate", po::value<float>()->notifier([&](const auto &v) { _learning_rate = v; }), 
      "Learning rate\n"
      "If not given, model's hyper parameter is used")
    ("loss", po::value<int>()
      ->notifier([&](const auto& v){_loss_type = checkValidation("loss", valid_loss, v);}),
      genHelpMsg("Loss type", valid_loss).c_str()
    )
    ("loss_reduction_type", po::value<int>()
      ->notifier([&](const auto &v){_loss_reduction_type = checkValidation("loss_reduction_type", valid_loss_rdt, v);}),
      genHelpMsg("Loss Reduction type", valid_loss_rdt).c_str()
    )
    ("optimizer", po::value<int>()
      ->notifier([&](const auto& v){_optimizer_type = checkValidation("optimizer", valid_optim, v);}),
      genHelpMsg("Optimizer type", valid_optim).c_str()
    )
    ("metric", po::value<int>()->default_value(-1)->notifier([&] (const auto &v) { _metric_type = v; }),
      "Metric type\n"
      "Simply calculates the metric value using the variables (default: none)\n"
      "0: CATEGORICAL_ACCURACY")
    ("validation_split", po::value<float>()->default_value(0.0f)->notifier(process_validation_split),
         "Float between 0 and 1(0 < float < 1). Fraction of the training data to be used as validation data.")
    ("verbose_level,v", po::value<int>()->default_value(0)->notifier([&](const auto &v) { _verbose_level = v; }),
         "Verbose level\n"
         "0: prints the only result. Messages btw run don't print\n"
         "1: prints result and message btw run\n"
         "2: prints all of messages to print")
    ("output_sizes", po::value<std::string>()->notifier(process_output_sizes),
        "The output buffer size in JSON 1D array\n"
        "If not given, the model's output sizes are used\n"
        "e.g. '[0, 40, 2, 80]' to set 0th tensor to 40 and 2nd tensor to 80.")
    ;
  // clang-format on

  _options.add(general);
  _positional.add("path", -1);
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);

  if (vm.count("help"))
  {
    std::cout << "onert_train\n\n";
    std::cout << "Usage: " << argv[0] << " [model path] [<options>]\n\n";
    std::cout << _options;
    std::cout << "\n";

    exit(0);
  }

  if (vm.count("version"))
  {
    _print_version = true;
    return;
  }

  {
    auto conflicting_options = [&](const std::string &o1, const std::string &o2) {
      if ((vm.count(o1) && !vm[o1].defaulted()) && (vm.count(o2) && !vm[o2].defaulted()))
      {
        throw boost::program_options::error(std::string("Two options '") + o1 + "' and '" + o2 +
                                            "' cannot be given at once.");
      }
    };

    // Cannot use both single model file and nnpackage at once
    conflicting_options("modelfile", "nnpackage");

    // Require modelfile, nnpackage, or path
    if (!vm.count("modelfile") && !vm.count("nnpackage") && !vm.count("path"))
      throw boost::program_options::error(
        std::string("Require one of options modelfile, nnpackage, or path."));
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
}

} // end of namespace onert_train
