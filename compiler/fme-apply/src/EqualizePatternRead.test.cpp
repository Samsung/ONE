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

#include "EqualizePatternRead.h"

#include <fstream>
#include <cstdio>

#include <gtest/gtest.h>

using namespace fme_apply;

namespace
{

void write_to_file(const std::string &filename, const std::string &content)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(content.c_str(), content.size()).fail())
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
}

class EqualizePatternReadTest : public ::testing::Test
{
public:
  EqualizePatternReadTest() { _filename = "test.json"; }

protected:
  virtual void SetUp() override { std::remove(_filename.c_str()); }

  virtual void TearDown() override { std::remove(_filename.c_str()); }

protected:
  std::string _filename;
};

} // namespace

TEST_F(EqualizePatternReadTest, simple)
{
  const std::string contents = "["
                               "{"
                               "\"front\": \"f\","
                               "\"back\": \"b\","
                               "\"type\": \"ScaleOnly\","
                               "\"act_scale\": [4.0, 5.0, 6.0]"
                               "}"
                               "]";

  EXPECT_NO_THROW(write_to_file(_filename, contents));

  auto patterns = fme_apply::read(_filename);

  EXPECT_EQ(1, patterns.size());
  EXPECT_EQ("f", patterns[0].front);
  EXPECT_EQ("b", patterns[0].back);
  EXPECT_EQ(EqualizePattern::Type::ScaleOnly, patterns[0].type);
  EXPECT_EQ(3, patterns[0].act_scale.size());
  EXPECT_FLOAT_EQ(4.0, patterns[0].act_scale[0]);
  EXPECT_FLOAT_EQ(5.0, patterns[0].act_scale[1]);
  EXPECT_FLOAT_EQ(6.0, patterns[0].act_scale[2]);
}

TEST_F(EqualizePatternReadTest, no_file_NEG) { EXPECT_ANY_THROW(fme_apply::read(_filename)); }

TEST_F(EqualizePatternReadTest, no_json_NEG)
{
  const std::string contents = "nojson";

  EXPECT_NO_THROW(write_to_file(_filename, contents));
  EXPECT_ANY_THROW(fme_apply::read(_filename));
}

TEST_F(EqualizePatternReadTest, no_front_NEG)
{
  const std::string contents = "["
                               "{"
                               "\"back\": \"b\","
                               "\"type\": \"ScaleOnly\","
                               "\"scale\": [1.0, 2.0, 3.0]"
                               "}"
                               "]";

  EXPECT_NO_THROW(write_to_file(_filename, contents));
  EXPECT_ANY_THROW(fme_apply::read(_filename));
}
