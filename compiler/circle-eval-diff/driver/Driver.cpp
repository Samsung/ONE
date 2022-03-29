/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleEvalDiff.h"

#include <arser/arser.h>
#include <vconone/vconone.h>

using namespace circle_eval_diff;

namespace
{

std::string to_lower_case(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

Metric str_to_metric(const std::string &str)
{
  if (to_lower_case(str).compare("mae") == 0)
    return Metric::MAE;

  throw std::runtime_error("Unsupported metric.");
}

InputFormat str_to_input_format(const std::string &str)
{
  if (to_lower_case(str).compare("h5") == 0)
    return InputFormat::H5;
  if (to_lower_case(str).compare("directory") == 0)
    return InputFormat::Directory;

  throw std::runtime_error("Unsupported input format.");
}

void print_version(void)
{
  std::cout << "circle-eval-diff version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

} // namespace

int entry(const int argc, char **argv)
{
  arser::Arser arser("Compare inference results of two circle models");

  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  arser.add_argument("-V", "--verbose")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("output additional information to stdout or stderr");

  arser.add_argument("--first_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("First input model filepath");

  arser.add_argument("--second_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Second input model filepath");

  arser.add_argument("--first_input_data")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Input data filepath for the first model. If not given, circle-eval-diff will run with "
          "randomly generated data");

  arser.add_argument("--second_input_data")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Input data filepath for the second model. If not given, circle-eval-diff will run with "
          "randomly generated data");

  arser.add_argument("--metric")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Metric for comparison (default: MAE)");

  arser.add_argument("--input_data_format")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Input data format. h5/hdf5 (default) or list/filelist");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  auto first_model_path = arser.get<std::string>("--first_model");
  auto second_model_path = arser.get<std::string>("--second_model");

  // Default values
  std::string first_input_data_path;
  std::string second_input_data_path;
  std::string metric("MAE");
  std::string input_data_format("h5");

  if (arser["--first_input_data"])
    first_input_data_path = arser.get<std::string>("--first_input_data");

  if (arser["--second_input_data"])
    second_input_data_path = arser.get<std::string>("--second_input_data");

  if ((not first_input_data_path.empty() and second_input_data_path.empty()) or
      (first_input_data_path.empty() and not second_input_data_path.empty()))
    throw std::runtime_error("Input data path should be given for both first_model and "
                             "second_model, or neither must be given.");

  bool input_data_given = not first_input_data_path.empty();

  if (arser["--metric"])
    metric = arser.get<std::string>("--metric");

  if (arser["--input_data_format"])
    input_data_format = arser.get<std::string>("--input_data_format");

  auto ctx = std::make_unique<CircleEvalDiff::Context>();
  {
    ctx->first_model_path = first_model_path;
    ctx->second_model_path = second_model_path;
    ctx->metric = str_to_metric(metric);
    ctx->input_format =
      input_data_given ? str_to_input_format(input_data_format) : InputFormat::Random;
  }

  CircleEvalDiff ced(std::move(ctx));

  ced.init();

  ced.evalDiff(first_input_data_path, second_input_data_path);

  return EXIT_SUCCESS;
}
