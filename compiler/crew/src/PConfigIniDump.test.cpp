/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "crew/PConfigIni.h"
#include "crew/PConfigIniDump.h"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

TEST(ConfigIniDumpTest, dump_sections)
{
  crew::Sections sections;
  crew::Section section;

  section.name = "hello";
  section.items["key"] = "value";

  sections.push_back(section);

  std::stringstream ss;

  ss << sections;

  // there's extra \n at end of each section
  ASSERT_TRUE(ss.str() == "[hello]\nkey=value\n\n");
}
