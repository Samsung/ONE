/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflchef/ModelChef.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <arser/arser.h>

#include <fstream>
#include <iostream>

int entry(int argc, char **argv)
{
  arser::Arser arser;
  arser.add_argument("recipe")
      .type(arser::DataType::STR)
      .help("Source recipe file path to convert");
  arser.add_argument("tflite").type(arser::DataType::STR).help("Target tflite file path");

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

  int32_t model_version = 1;

  ::tflchef::ModelRecipe model_recipe;

  std::string recipe_path = arser.get<std::string>("recipe");
  // Load model recipe from a file
  {
    std::ifstream is{recipe_path};
    google::protobuf::io::IstreamInputStream iis{&is};
    if (!google::protobuf::TextFormat::Parse(&iis, &model_recipe))
    {
      std::cerr << "ERROR: Failed to parse recipe '" << recipe_path << "'" << std::endl;
      return 255;
    }

    if (model_recipe.has_version())
    {
      model_version = model_recipe.version();
    }
  }

  if (model_version > 1)
  {
    std::cerr << "ERROR: Unsupported recipe version: " << model_version << ", '" << argv[1] << "'"
              << std::endl;
    return 255;
  }

  auto generated_model = tflchef::cook(model_recipe);

  std::string tflite_path = arser.get<std::string>("tflite");
  // Dump generated model into a file
  {
    std::ofstream os{tflite_path, std::ios::binary};
    os.write(generated_model.base(), generated_model.size());
  }

  return 0;
}
