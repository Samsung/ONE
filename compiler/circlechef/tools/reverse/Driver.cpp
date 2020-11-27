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

#include <circlechef/RecipeChef.h>

#include <arser/arser.h>
#include <foder/FileLoader.h>

#include <memory>
#include <iostream>

int entry(int argc, char **argv)
{
  arser::Arser arser;
  arser.add_argument("circle")
    .type(arser::DataType::STR)
    .help("Source circle file path to convert");
  arser.add_argument("recipe").type(arser::DataType::STR).help("Target recipe file path");

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

  std::string circle_path = arser.get<std::string>("circle");
  // Load TF lite model from a circle file
  const foder::FileLoader fileLoader{circle_path};
  std::vector<char> modelData = fileLoader.load();
  const circle::Model *circlemodel = circle::GetModel(modelData.data());
  if (circlemodel == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << circle_path << "'" << std::endl;
    return 255;
  }

  // Generate ModelRecipe recipe
  std::unique_ptr<circlechef::ModelRecipe> recipe = circlechef::generate_recipe(circlemodel);
  if (recipe.get() == nullptr)
  {
    std::cerr << "ERROR: Failed to generate recipe" << std::endl;
    return 255;
  }

  std::string recipe_path = arser.get<std::string>("recipe");
  // Save to a file
  bool result = circlechef::write_recipe(recipe_path, recipe);
  if (!result)
  {
    std::cerr << "ERROR: Failed to write to recipe '" << recipe_path << "'" << std::endl;
    return 255;
  }
  return 0;
}
