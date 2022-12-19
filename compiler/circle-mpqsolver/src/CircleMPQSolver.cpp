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

#include <iostream>

void print_version(void)
{
  std::cout << "circle-mpqsolver version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  const std::string bisection_str = "--bisection";

  arser::Arser arser("circle-mpqsolver provides light-weight methods for finding a high-quality "
                     "mixed-precision model within a reasonable time.");

  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);

  arser.add_argument("--data").required(true).help("Path to the test data");
  arser.add_argument("--data_format").required(false).help("Test data format (default: h5)");

  arser.add_argument("--qerror_ratio")
    .type(arser::DataType::FLOAT)
    .default_value(0.5f)
    .help("quantization error ratio (> 0 and < 1)");

  arser.add_argument(bisection_str)
    .nargs(1)
    .default_value("auto")
    .type(arser::DataType::STR)
    .help("Single optional argument for bisection method. "
          "Whether input node should be quantized to Q16: 'auto', 'true', 'false'.");

  arser.add_argument("--input_model")
    .required(true)
    .help("Input float model with min max initialized");

  arser.add_argument("--output_model").required(true).help("Output quantized model");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  // TODO Run Bisect method
  return EXIT_SUCCESS;
}
