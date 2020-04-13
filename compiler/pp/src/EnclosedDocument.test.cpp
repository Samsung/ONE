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

#include "pp/EnclosedDocument.h"

#include <gtest/gtest.h>

TEST(LINEAR_DOCUMENT, lines)
{
  pp::EnclosedDocument doc;

  doc.front().append("A");
  doc.back().append("C");
  doc.back().append("B");

  ASSERT_EQ(doc.lines(), 3);
}

TEST(LINEAR_DOCUMENT, line)
{
  pp::EnclosedDocument doc;

  doc.front().append("A");
  doc.front().indent();
  doc.front().append("B");
  doc.back().append("C");
  doc.back().append("B");

  ASSERT_EQ(doc.lines(), 4);
  ASSERT_EQ(doc.line(0), "A");
  ASSERT_EQ(doc.line(1), "  B");
  ASSERT_EQ(doc.line(2), "B");
  ASSERT_EQ(doc.line(3), "C");
}
