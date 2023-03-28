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

#include <json.h>
#include <fstream>
#include <gtest/gtest.h>

namespace
{

void writeDataToFile(const std::string &path, const std::string &data)
{
  std::ofstream file;
  file.open(path);
  file << data;
  file.close();
}

} // namespace

TEST(CircleMPQSolverVISQErrorApproximatorTest, verifyResultsTest)
{
  static std::string errors_key = "error";
  static std::string layer_key = "layer_0";
  static float layer_error = 0.5f;
  // trivial json with a single layer
  Json::Value error_data;
  Json::Value layer_data;
  layer_data[layer_key] = layer_error;
  error_data[errors_key].append(layer_data);

  Json::StreamWriterBuilder builder;
  auto data = Json::writeString(builder, error_data);
  std::string path = std::tmpnam(nullptr);
  writeDataToFile(path, data);

  mpqsolver::bisection::VISQErrorApproximator approximator;
  EXPECT_NO_THROW(approximator.init(path));
  EXPECT_FLOAT_EQ(approximator.approximate(layer_key), layer_error);
  unlink(path.c_str());
}

TEST(CircleMPQSolverVISQErrorApproximatorTest, verifyResultsTest_NEG)
{
  Json::Value error_data;
  // just an empty json
  Json::StreamWriterBuilder builder;
  auto data = Json::writeString(builder, error_data);
  std::string path = std::tmpnam(nullptr);
  writeDataToFile(path, data);

  mpqsolver::bisection::VISQErrorApproximator approximator;
  EXPECT_THROW(approximator.init(path), std::exception);
  unlink(path.c_str());
}
