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
#include <unistd.h>
#include <numeric>
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
T checkValidation(const std::string &arg_name, const std::vector<T> &valid_args, uint32_t value)
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
  _arser.add_argument("path").type(arser::DataType::STR).help("NN Package or NN Modelfile path");

  _arser.add_argument("--version")
    .nargs(0)
    .default_value(false)
    .help("Print version and exit immediately");
  _arser.add_argument("--checkpoint").type(arser::DataType::STR).help("Checkpoint filename");
  _arser.add_argument("--export_circle").type(arser::DataType::STR).help("Path to export circle");
  _arser.add_argument("--export_circleplus")
    .type(arser::DataType::STR)
    .help("Path to export circle+");
  _arser.add_argument("--export_checkpoint")
    .type(arser::DataType::STR)
    .help("Path to export checkpoint");
  _arser.add_argument("--load_input:raw")
    .type(arser::DataType::STR)
    .help({"NN Model Raw Input data file", "The datafile must have data for each input number.",
           "If there are 3 inputs, the data of input0 must exist as much as data_length,",
           "and the data for input1 and input2 must be held sequentially as data_length."});
  _arser.add_argument("--load_expected:raw")
    .type(arser::DataType::STR)
    .help({"NN Model Raw Expected data file", "(Same data policy with load_input:raw)"});
  _arser.add_argument("--mem_poll", "-m")
    .nargs(0)
    .default_value(false)
    .help("Check memory polling (default: false)");
  _arser.add_argument("--epoch")
    .type(arser::DataType::INT32)
    .default_value(5)
    .help("Epoch number (default: 5)");
  _arser.add_argument("--batch_size")
    .type(arser::DataType::INT32)
    .help({"Batch size", "If not given, model's hyper parameter is used"});
  _arser.add_argument("--learning_rate")
    .type(arser::DataType::FLOAT)
    .help({"Learning rate", "If not given, model's hyper parameter is used"});
  _arser.add_argument("--loss").type(arser::DataType::INT32).help("Loss type");
  _arser.add_argument("--loss_reduction_type")
    .type(arser::DataType::INT32)
    .help("Loss reduction type");
  _arser.add_argument("--optimizer").type(arser::DataType::INT32).help("Optimizer type");
  _arser.add_argument("--metric")
    .type(arser::DataType::INT32)
    .default_value(-1)
    .help({"Metric type", "Simply calculates the metric value using the variables (default: none)",
           "0: CATEGORICAL_ACCURACY"});
  _arser.add_argument("--validation_split")
    .type(arser::DataType::FLOAT)
    .default_value(0.0f)
    .help("Float between 0 and 1(0 < float < 1). Fraction of the training data to be used as "
          "validation data.");
  _arser.add_argument("--verbose_level", "-v")
    .type(arser::DataType::INT32)
    .default_value(0)
    .help({"Verbose level", "0: prints the only result. Messages btw run don't print",
           "1: prints result and message btw run", "2: prints all of messages to print"});
  _arser.add_argument("--output_sizes")
    .type(arser::DataType::STR)
    .help({"The output buffer size in JSON 1D array",
           "If not given, the model's output sizes are used",
           "e.g. '[0, 40, 2, 80]' to set 0th tensor to 40 and 2nd tensor to 80."});
  _arser.add_argument("--num_of_trainable_ops")
    .type(arser::DataType::INT32)
    .help({"Number of the layers to be trained from the back of the model.",
           "\"-1\" means that all layers will be trained.",
           "\"0\" means that no layer will be trained."});
}

void Args::Parse(const int argc, char **argv)
{
  try
  {
    _arser.parse(argc, argv);

    if (_arser.get<bool>("--version"))
    {
      _print_version = true;
      return;
    }

    if (_arser["path"])
    {
      auto path = _arser.get<std::string>("path");
      struct stat sb;
      if (stat(path.c_str(), &sb) == 0)
      {
        if (sb.st_mode & S_IFDIR)
        {
          _package_filename = path;
          checkPackage(path);
          std::cout << "Package Filename " << path << std::endl;
        }
        else
        {
          _model_filename = path;
          checkModelfile(path);
          std::cout << "Model Filename " << path << std::endl;
          _use_single_model = true;
        }
      }
      else
      {
        std::cerr << "Cannot find: " << path << "\n";
        exit(1);
      }
    }

    if (_arser["--checkpoint"])
    {
      _checkpoint_filename = _arser.get<std::string>("--checkpoint");
      checkModelfile(_checkpoint_filename);
    }

    if (_arser["--export_circle"])
      _export_circle_filename = _arser.get<std::string>("--export_circle");
    if (_arser["--export_circleplus"])
      _export_circleplus_filename = _arser.get<std::string>("--export_circleplus");
    if (_arser["--export_checkpoint"])
      _export_checkpoint_filename = _arser.get<std::string>("--export_checkpoint");
    if (_arser["--load_input:raw"])
    {
      _load_raw_input_filename = _arser.get<std::string>("--load_input:raw");
      checkModelfile(_load_raw_input_filename);
    }
    if (_arser["--load_expected:raw"])
    {
      _load_raw_expected_filename = _arser.get<std::string>("--load_expected:raw");
      checkModelfile(_load_raw_expected_filename);
    }

    _mem_poll = _arser.get<bool>("--mem_poll");
    _epoch = _arser.get<int32_t>("--epoch");

    if (_arser["--batch_size"])
      _batch_size = _arser.get<int32_t>("--batch_size");
    if (_arser["--learning_rate"])
      _learning_rate = _arser.get<float>("--learning_rate");
    if (_arser["--loss"])
      _loss_type = checkValidation("loss", valid_loss, _arser.get<int32_t>("--loss"));
    if (_arser["--loss_reduction_type"])
      _loss_reduction_type = checkValidation("loss_reduction_type", valid_loss_rdt,
                                             _arser.get<int>("--loss_reduction_type"));
    if (_arser["--optimizer"])
      _optimizer_type = checkValidation("optimizer", valid_optim, _arser.get<int>("--optimizer"));
    _metric_type = _arser.get<int32_t>("--metric");

    _validation_split = _arser.get<float>("--validation_split");
    if (_validation_split < 0.f || _validation_split > 1.f)
    {
      std::cerr << "Invalid validation_split. Float between 0 and 1." << std::endl;
      exit(1);
    }

    _verbose_level = _arser.get<int32_t>("--verbose_level");

    if (_arser["--output_sizes"])
    {
      auto output_sizes_json_str = _arser.get<std::string>("--output_sizes");
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

    if (_arser["--num_of_trainable_ops"])
      _num_of_trainable_ops = _arser.get<int32_t>("--num_of_trainable_ops");
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "Bad cast error - " << e.what() << '\n';
    exit(1);
  }
}

} // end of namespace onert_train
