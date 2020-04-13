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

#include <tflread/Model.h>
#include <tfldump/Dump.h>

#include <iostream>

int entry(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << "USAGE: " << argv[0] << " [tflite]" << std::endl;
    return 255;
  }

  // Load TF lite model from a tflite file
  std::unique_ptr<tflread::Model> model = tflread::load_tflite(argv[1]);
  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load tflite '" << argv[1] << "'" << std::endl;
    return 255;
  }

  const tflite::Model *tflmodel = model->model();
  if (tflmodel == nullptr)
  {
    std::cerr << "ERROR: Failed to load tflite '" << argv[1] << "'" << std::endl;
    return 255;
  }

  std::cout << "Dump: " << argv[1] << std::endl << std::endl;

  std::cout << tflmodel << std::endl;

  return 0;
}
