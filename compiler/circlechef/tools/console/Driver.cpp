/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "circlechef/ModelChef.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>

int entry(int argc, char **argv)
{
  int32_t model_version = 1;

  ::circlechef::ModelRecipe model_recipe;

  // Read a model recipe from standard input
  {
    google::protobuf::io::IstreamInputStream iis{&std::cin};
    if (!google::protobuf::TextFormat::Parse(&iis, &model_recipe))
    {
      std::cerr << "ERROR: Failed to parse recipe" << std::endl;
      return 255;
    }

    if (model_recipe.has_version())
    {
      model_version = model_recipe.version();
    }
  }

  if (model_version > 1)
  {
    std::cerr << "ERROR: Unsupported recipe version: " << model_version << std::endl;
    return 255;
  }

  auto generated_model = circlechef::cook(model_recipe);

  // Write a generated model into standard output
  std::cout.write(generated_model.base(), generated_model.size());

  return 0;
}
