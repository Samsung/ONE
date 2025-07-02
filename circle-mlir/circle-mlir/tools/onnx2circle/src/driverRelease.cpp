/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "onnx2circle.h"
#include "cmdOptions.h"

#include <arser/arser.h>

#include <iostream>

using namespace opts;

std::string get_copyright(void)
{
  std::string str;
  str = "Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved\r\n";
  str += "Licensed under the Apache License, Version 2.0\r\n";
  str += "https://github.com/Samsung/ONE";
  return str;
}

void print_version(void)
{
  std::cout << "onnx2circle version " << __version << std::endl;
  std::cout << get_copyright() << std::endl;
}

void print_version_only(void) { std::cout << __version; }

int safe_main(int argc, char *argv[])
{
  arser::Arser arser;

  arser::Helper::add_version(arser, print_version);

  arser.add_argument("--version_only")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version number only and exit")
    .exit_with(print_version_only);

  arser.add_argument("--save_ops")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_save_ops);

  arser.add_argument("--dynamic_batch_to_single_batch")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_dynamic_batch_to_single_batch);

  arser.add_argument("--unroll_rnn")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_unroll_rnn_d);

  arser.add_argument("--unroll_lstm")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_unroll_lstm_d);

  arser.add_argument("--experimental_disable_batchmatmul_unfold")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_edbuf_d);

  // ignored obsolete options
  arser.add_argument("--keep_io_order")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_keep_io_order_d);

  arser.add_argument("--save_intermediate")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_save_int_d);

  arser.add_argument("--check_shapeinf")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_check_shapeinf);

  arser.add_argument("--check_dynshapeinf")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_check_dynshapeinf);

  arser.add_argument("--check_rawprocessed")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help(__opt_check_rawprocessed);

  // two positional arguments
  arser.add_argument("onnx").help("Input ONNX file");
  arser.add_argument("circle").help("Output Circle file");

  arser.parse(argc, argv);

  O2Cparam param;
  param.sourcefile = arser.get<std::string>("onnx");
  param.targetfile = arser.get<std::string>("circle");
  param.save_ops = arser.get<bool>("--save_ops");
  param.dynamic_batch_to_single_batch = arser.get<bool>("--dynamic_batch_to_single_batch");
  param.unroll_rnn = arser.get<bool>("--unroll_rnn");
  param.unroll_lstm = arser.get<bool>("--unroll_lstm");
  param.unfold_batchmatmul = !arser.get<bool>("--experimental_disable_batchmatmul_unfold");
  param.check_shapeinf = arser.get<bool>("--check_shapeinf");
  param.check_dynshapeinf = arser.get<bool>("--check_dynshapeinf");
  param.check_rawprocessed = arser.get<bool>("--check_rawprocessed");

  return entry(param);
}

int main(int argc, char *argv[])
{
  try
  {
    return safe_main(argc, argv);
  }
  catch (const std::exception &err)
  {
    std::cout << err.what() << '\n';
  }
  return -1;
}
