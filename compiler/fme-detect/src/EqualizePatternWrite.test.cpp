/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "EqualizePatternWrite.h"

#include <fstream>
#include <cstdio>

#include <json.h>
#include <gtest/gtest.h>

using namespace fme_detect;

namespace
{

Json::Value read_json(const std::string &filename)
{
  Json::Value root;
  std::ifstream ifs(filename);

  // Failed to open cfg file
  if (not ifs.is_open())
    throw std::runtime_error("Cannot open config file. " + filename);

  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;

  // Failed to parse
  if (not parseFromStream(builder, ifs, &root, &errs))
    throw std::runtime_error("Cannot parse config file (json format). " + errs);

  return root;
}

class EqualizePatternWriteTest : public ::testing::Test
{
public:
  EqualizePatternWriteTest() { _filename = "test.json"; }

protected:
  virtual void SetUp() override { std::remove(_filename.c_str()); }

  virtual void TearDown() override { std::remove(_filename.c_str()); }

protected:
  std::string _filename;
};

} // namespace

TEST_F(EqualizePatternWriteTest, empty_pattern)
{
  std::vector<EqualizePattern> p;
  EXPECT_NO_THROW(fme_detect::write(p, _filename));

  auto root = read_json(_filename);

  EXPECT_TRUE(root.isArray());
  EXPECT_TRUE(root.empty());
}

TEST_F(EqualizePatternWriteTest, empty_name_NEG)
{
  std::vector<EqualizePattern> p;
  EXPECT_ANY_THROW(fme_detect::write(p, ""));
}

TEST_F(EqualizePatternWriteTest, simple)
{
  std::vector<EqualizePattern> p;
  {
    p.emplace_back("f", "b", EqualizePattern::Type::ScaleOnly);
  }
  EXPECT_NO_THROW(fme_detect::write(p, _filename));

  auto root = read_json(_filename);

  EXPECT_TRUE(root.isArray());
  EXPECT_EQ("f", root[0]["front"].asString());
  EXPECT_EQ("b", root[0]["back"].asString());
  EXPECT_EQ("ScaleOnly", root[0]["type"].asString());
}
