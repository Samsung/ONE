/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "VISQErrorApproximator.h"

#include <fstream>
#include <json.h>

using namespace mpqsolver::bisection;

void VISQErrorApproximator::init(const std::string &visq_data_path)
{
  // read file
  std::ifstream file(visq_data_path);
  init(file);
}

void VISQErrorApproximator::init(std::istream &visq_data)
{
  Json::Reader reader;
  Json::Value completeJsonData;
  if (!reader.parse(visq_data, completeJsonData))
  {
    throw std::runtime_error("Invalid visq stream");
  }

  if (!completeJsonData.isMember("error"))
  {
    throw std::runtime_error("No 'error' section in visq stream");
  }

  auto layers = completeJsonData["error"][0];
  auto names = layers.getMemberNames();
  for (const auto &name : names)
  {
    auto value = layers[name].asFloat();
    _layer_errors[name] = value;
  }
}

float VISQErrorApproximator::approximate(const std::string &node_name) const
{
  auto iter = _layer_errors.find(node_name);
  if (iter == _layer_errors.end())
  {
    return 0.f;
  }

  return iter->second;
}
