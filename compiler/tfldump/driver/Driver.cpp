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

#include <arser/arser.h>
#include <foder/FileLoader.h>
#include <tfldump/Dump.h>

#include <iostream>
#include <cstdint>

int entry(int argc, char **argv)
{
  arser::Arser arser;
  arser.add_argument("tflite").help("TFLite file to dump");

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

  std::string tflite_path = arser.get<std::string>("tflite");
  // Load TF lite model from a tflite file
  foder::FileLoader fileLoader{tflite_path};
  std::vector<char> modelData = fileLoader.load();
  const tflite::Model *tflmodel = tflite::GetModel(modelData.data());
  if (tflmodel == nullptr)
  {
    std::cerr << "ERROR: Failed to load tflite '" << tflite_path << "'" << std::endl;
    return 255;
  }

  std::cout << "Dump: " << tflite_path << std::endl << std::endl;

  std::cout << tflmodel << std::endl;

  return 0;
}
