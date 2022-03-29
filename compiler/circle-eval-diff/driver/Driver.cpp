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

#include <arser/arser.h>
#include <vconone/vconone.h>

namespace
{

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
    .help("Input data format. h5/hdf5 (default) or directory");

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

  // TODO Run CircleEvalDiff

  return EXIT_SUCCESS;
}
