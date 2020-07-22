/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "VerifyFlatBuffers.h"

#include <arser/arser.h>

#include <iostream>
#include <memory>
#include <string>

int entry(int argc, char **argv)
{
  arser::Arser arser;
  arser.add_argument("circle").type(arser::DataType::STR).help("Circle file path to verify");

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

  auto verifier = std::make_unique<VerifyFlatbuffers>();

  std::string model_file = arser.get<std::string>("circle");

  std::cout << "[ RUN       ] Check " << model_file << std::endl;

  auto result = verifier->run(model_file);

  if (result == 0)
  {
    std::cout << "[      PASS ] Check " << model_file << std::endl;
  }
  else
  {
    std::cout << "[      FAIL ] Check " << model_file << std::endl;
  }

  return result;
}
