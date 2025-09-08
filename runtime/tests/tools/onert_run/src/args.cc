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
#include "nnfw_util.h"

#include <functional>
#include <unistd.h>
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

// param shape_str is a form of, e.g., "[1, [2, 3], 3, []]" or "h5"
void handleShapeJsonParam(onert_run::TensorShapeMap &shape_map, const std::string &shape_str)
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

} // namespace

namespace onert_run
{

Args::Args(const int argc, char **argv)
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  _arser.add_argument("path").type(arser::DataType::STR).help("NN Package or NN Modelfile path");

  arser::Helper::add_version(_arser, print_version);
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
  _arser.add_argument("--dump", "-d").type(arser::DataType::STR).help("Output filename");
  _arser.add_argument("--load", "-l").type(arser::DataType::STR).help("Input filename");
#endif
  _arser.add_argument("--dump:raw").type(arser::DataType::STR).help("Raw Output filename");
  _arser.add_argument("--dump_input:raw")
    .type(arser::DataType::STR)
    .help("Raw Input filename for dump");
  _arser.add_argument("--load:raw").type(arser::DataType::STR).help("Raw Input filename");
  _arser.add_argument("--output_sizes")
    .type(arser::DataType::STR)
    .help({"The output buffer size in JSON 1D array",
           "If not given, the model's output sizes are used",
           "e.g. '[0, 40, 2, 80]' to set 0th tensor to 40 and 2nd tensor to 80."});
  _arser.add_argument("--num_runs", "-r")
    .type(arser::DataType::INT32)
    .default_value(1)
    .help("The number of runs");
  _arser.add_argument("--fixed_input")
    .nargs(0)
    .default_value(false)
    .help("Use same random input data on each run (avaliable on random input)");
  _arser.add_argument("--force_float")
    .nargs(0)
    .default_value(false)
    .help("Ignore model's input and output type and use float type buffer");
  _arser.add_argument("--warmup_runs", "-w")
    .type(arser::DataType::INT32)
    .default_value(0)
    .help("The number of warmup runs");
  _arser.add_argument("--minmax_runs")
    .type(arser::DataType::INT32)
    .default_value(0)
    .help("The number of minmax recording runs before full quantization");
  _arser.add_argument("--run_delay", "--t")
    .type(arser::DataType::INT32)
    .default_value(-1)
    .help("Delay time(us) between runs (as default no delay)");
  _arser.add_argument("--gpumem_poll", "-g")
    .nargs(0)
    .default_value(false)
    .help("Check gpu memory polling separately");
  _arser.add_argument("--mem_poll", "-m")
    .nargs(0)
    .default_value(false)
    .help("Check memory polling");
  _arser.add_argument("--write_report", "-p")
    .nargs(0)
    .default_value(false)
    .help({"Write report", "{exec}-{nnpkg|modelfile}-{backend}.csv will be generated.",
           "e.g. onert_run-UNIT_Add_000-acl_cl.csv.",
           "{nnpkg|modelfile} name may be changed to realpath if you use symbolic-link."});
  _arser.add_argument("--shape_prepare")
    .type(arser::DataType::STR)
    .default_value("")
    .help("Please refer to the description of 'shape_run'");
  _arser.add_argument("--shape_run").type(arser::DataType::STR).default_value("").help({
    "--shape_prepare: set shape of tensors before compilation (before calling nnfw_prepare()).",
      "--shape_run: set shape of tensors before running (before calling nnfw_run()).",
      "Allowed value:.",
      "'[0, [1, 2], 2, []]': set 0th tensor to [1, 2] and 2nd tensor to [] (scalar).",
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
      "'h5': read shape(s) from H5 input file. '--load' should also be provided.",
      "if '--load' option is provided but '--shape_prepare' or '--shape_run' is not provided,",
      "'--shape_run h5' will be used by default.",
#endif
      "For detailed description, please consutl the description of nnfw_set_input_tensorinfo()"
  });
  _arser.add_argument("--output_shape")
    .type(arser::DataType::STR)
    .default_value("")
    .help({"Set output shape for dump. Size should be same.",
           "'[0, [1, 2], 2, []]': set 0th tensor to [1, 2] and 2nd tensor to [] (scalar)."});
  _arser.add_argument("--verbose_level", "-v")
    .type(arser::DataType::INT32)
    .default_value(0)
    .help({"Verbose level", "0: prints the only result. Messages btw run don't print",
           "1: prints result and message btw run", "2: prints all of messages to print"});
  _arser.add_argument("--quantize", "-q")
    .type(arser::DataType::STR)
    .help({"Request quantization with type", "uint8, int16: full quantization",
           "int8_wo, int16_wo: weight only quantization"});
  _arser.add_argument("--qpath")
    .type(arser::DataType::STR)
    .help({"Path to export quantized model.",
           "If it is not set, the quantized model will be exported to the same directory of the "
           "original model/package with q8/q16 suffix."});
  _arser.add_argument("--codegen", "-c")
    .type(arser::DataType::STR)
    .help({"Target backend name for code generation",
           "The target string will be used to find a backend library.",
           "This string should be in the following format:", "{backend extension} + '-gen'.",
           "For detailed description, please see the description of nnfw_codegen()"});
  _arser.add_argument("--cpath")
    .type(arser::DataType::STR)
    .help({"Path to export target-dependent model.",
           "If it is not set, the generated model will be exported to the same directory of the "
           "original model/package with target backend extension."});
  _arser.add_argument("--signature")
    .type(arser::DataType::STR)
    .help({"Signature to select.", "If it is not set, 0th subgraph will be selected"});
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

#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
    if (_arser["--dump"])
      _dump_filename = _arser.get<std::string>("--dump");
    if (_arser["--load"])
      _load_filename = _arser.get<std::string>("--load");
#endif

    if (_arser["--dump:raw"])
      _dump_raw_filename = _arser.get<std::string>("--dump:raw");
    if (_arser["--dump_input:raw"])
      _dump_raw_input_filename = _arser.get<std::string>("--dump_input:raw");
    if (_arser["--load:raw"])
      _load_raw_filename = _arser.get<std::string>("--load:raw");

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

    _num_runs = _arser.get<int32_t>("--num_runs");
    _fixed_input = _arser.get<bool>("--fixed_input");
    _force_float = _arser.get<bool>("--force_float");
    _warmup_runs = _arser.get<int32_t>("--warmup_runs");
    _minmax_runs = _arser.get<int32_t>("--minmax_runs");
    _run_delay = _arser.get<int32_t>("--run_delay");
    _gpumem_poll = _arser.get<bool>("--gpumem_poll");
    _mem_poll = _arser.get<bool>("--mem_poll");
    _write_report = _arser.get<bool>("--write_report");

    auto shape_prepare = _arser.get<std::string>("--shape_prepare");
    auto shape_run = _arser.get<std::string>("--shape_run");
    if (!shape_prepare.empty())
    {
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
      if (shape_prepare == "H5" || shape_prepare == "h5")
      {
        _when_to_use_h5_shape = WhenToUseH5Shape::PREPARE;
        return;
      }
#endif
      try
      {
        handleShapeJsonParam(_shape_prepare, shape_prepare);
      }
      catch (const std::exception &e)
      {
        std::cerr << "error with '--shape_prepare' option: " << shape_prepare << std::endl;
        exit(1);
      }
    }

    if (!shape_run.empty())
    {
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
      if (shape_run == "H5" || shape_run == "h5")
      {
        _when_to_use_h5_shape = WhenToUseH5Shape::RUN;
        return;
      }
#endif
      try
      {
        handleShapeJsonParam(_shape_run, shape_run);
      }
      catch (const std::exception &e)
      {
        std::cerr << "error with '--shape_run' option: " << shape_run << std::endl;
        exit(1);
      }
    }

    auto output_shape = _arser.get<std::string>("--output_shape");
    if (!output_shape.empty())
    {
      try
      {
        handleShapeJsonParam(_output_shape, output_shape);
      }
      catch (const std::exception &e)
      {
        std::cerr << "error with '--output_shape' option: " << output_shape << std::endl;
        exit(1);
      }
    }

    _verbose_level = _arser.get<int32_t>("--verbose_level");

    if (_arser["--quantize"])
      _quantize = _arser.get<std::string>("--quantize");
    if (_arser["--qpath"])
      _quantized_model_path = _arser.get<std::string>("--qpath");
    if (_arser["--codegen"])
      _codegen = _arser.get<std::string>("--codegen");
    if (_arser["--cpath"])
      _codegen_model_path = _arser.get<std::string>("--cpath");
    if (_arser["--signature"])
      _signature = _arser.get<std::string>("--signature");

    // This must be run after parsing as `_warm_up_runs` must have been processed before.
    // Instead of EXECUTE to avoid overhead, memory polling runs on WARMUP
    if (_mem_poll && _warmup_runs == 0)
    {
      _warmup_runs = 1;
    }
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "Bad cast error - " << e.what() << '\n';
    exit(1);
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

} // end of namespace onert_run
