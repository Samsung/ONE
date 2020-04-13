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

#include "pp/MultiLineTextUtils.h"

#include <sstream>
#include <vector>

#include <gtest/gtest.h>

struct DummyMultiLineText final : public pp::MultiLineText
{
  std::vector<std::string> content;

  uint32_t lines(void) const override { return content.size(); }
  const std::string &line(uint32_t n) const override { return content.at(n); }
};

TEST(MUILTI_LINE_TEXT_UTILS, operator_shift)
{
  DummyMultiLineText txt;

  txt.content.emplace_back("A");
  txt.content.emplace_back("  B");
  txt.content.emplace_back("    C");

  const char *expected = "A\n"
                         "  B\n"
                         "    C\n";

  std::stringstream ss;

  ss << txt << std::endl;

  ASSERT_EQ(ss.str(), expected);
}
