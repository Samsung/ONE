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

#include <iostream>
#include <vector>

#include "CircleModel.h"
#include "TFLModel.h"

int entry(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << "USAGE: " << argv[0] << " [tflite] [circle]" << std::endl;
    return 255;
  }

  // read tflite file
  tflite2circle::TFLModel tfl_model(argv[1]);
  if (!tfl_model.is_valid())
  {
    std::cerr << "ERROR: Failed to load tflite '" << argv[1] << "'" << std::endl;
    return 255;
  }

  // create flatbuffer builder
  auto flatbuffer_builder = stdex::make_unique<flatbuffers::FlatBufferBuilder>(1024);

  // convert tflite to circle
  tflite2circle::CircleModel circle_model{flatbuffer_builder, tfl_model};

  std::ofstream outfile{argv[2], std::ios::binary};

  outfile.write(circle_model.base(), circle_model.size());
  outfile.close();
  // TODO find a better way of error handling
  if (outfile.fail())
  {
    std::cerr << "ERROR: Failed to write circle '" << argv[1] << "'" << std::endl;
    return 255;
  }

  return 0;
}
