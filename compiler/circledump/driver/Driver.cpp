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

#include <arser/arser.h>
#include <circleread/Model.h>
#include <circledump/Dump.h>

#include <iostream>

int entry(int argc, char **argv)
{
  arser::Arser arser;
  arser.add_argument("circle").type(arser::DataType::STR).help("Circle file path to dump");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << '\n';
    std::cout << arser;
    return 255;
  }

  std::string circle_path = arser.get<std::string>("circle");
  // Load Circle model from a circle file
  std::unique_ptr<circleread::Model> model = circleread::load_circle(circle_path);
  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << circle_path << "'" << std::endl;
    return 255;
  }

  const circle::Model *circlemodel = model->model();
  if (circlemodel == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << circle_path << "'" << std::endl;
    return 255;
  }

  std::cout << "Dump: " << circle_path << std::endl << std::endl;

  std::cout << circlemodel << std::endl;

  return 0;
}
