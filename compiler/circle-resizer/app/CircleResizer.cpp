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

#include "ModelEditor.h"
#include "ShapeParser.h"

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>
#include <sstream>
#include <memory>

using namespace circle_resizer;

namespace
{

void print_version()
{
  std::cout << "circle-resizer version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

} // namespace

int entry(const int argc, char **argv)
{
  arser::Arser arser("circle-resizer provides capabilities to change inputs of the models");

  arser.add_argument("--input_path")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("The path to the input model (.circle)");

  arser.add_argument("--output_path")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("The path to the resized model (.circle)");

  arser.add_argument("--input_shapes")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("New inputs shapes in in comma separated format. An example for 2 inputs: [1,2],[3,4].");

  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  try
  {
    arser.parse(argc, argv);

    const auto input_path = arser.get<std::string>("--input_path");

    auto circle_model = std::make_shared<CircleModel>(input_path);
    ModelEditor resizer(circle_model);
    const auto input_shapes = circle_model->input_shapes();
    std::cout << "Input shapes before resizing:" << std::endl;
    for (size_t in_idx = 0; in_idx < input_shapes.size(); ++in_idx)
    {
      std::cout << in_idx + 1 << ". " << input_shapes[in_idx] << std::endl;
    }

    auto output_shapes = circle_model->output_shapes();
    std::cout << "Output shapes before resizing:" << std::endl;
    for (size_t out_idx = 0; out_idx < output_shapes.size(); ++out_idx)
    {
      std::cout << out_idx + 1 << ". " << output_shapes[out_idx] << std::endl;
    }

    const auto output_path = arser.get<std::string>("--output_path");
    const auto new_input_shapes_str = arser.get<std::string>("--input_shapes");

    resizer.resize_inputs(parse_shapes(new_input_shapes_str));

    output_shapes = circle_model->output_shapes();
    std::cout << "Output shapes after resizing:" << std::endl;
    for (size_t out_idx = 0; out_idx < output_shapes.size(); ++out_idx)
    {
      std::cout << out_idx + 1 << ". " << output_shapes[out_idx] << std::endl;
    }

    circle_model->save(output_path);
    std::cout << "Resizing complete, the model saved to: " << output_path << std::endl;
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
