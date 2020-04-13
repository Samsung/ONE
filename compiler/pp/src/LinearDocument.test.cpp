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

#include "pp/LinearDocument.h"

#include <gtest/gtest.h>

TEST(LINEAR_DOCUMENT, append_void)
{
  pp::LinearDocument doc;

  doc.indent();
  doc.append();

  ASSERT_EQ(doc.lines(), 1);
  ASSERT_EQ(doc.line(0), "");
}

TEST(LINEAR_DOCUMENT, append_empty_string)
{
  pp::LinearDocument doc;

  doc.indent();
  doc.append("");

  ASSERT_EQ(doc.lines(), 1);
  ASSERT_EQ(doc.line(0), "");
}

TEST(LINEAR_DOCUMENT, formatted_append)
{
  pp::LinearDocument doc;

  doc.append("Hello ", 1);
  ASSERT_EQ(doc.lines(), 1);
  ASSERT_EQ(doc.line(0), "Hello 1");
}

TEST(LINEAR_DOCUMENT, forward_append)
{
  pp::LinearDocument doc;

  ASSERT_EQ(doc.lines(), 0);

  doc.append("A");
  doc.append("B");
  doc.append("C");

  ASSERT_EQ(doc.lines(), 3);
  ASSERT_EQ(doc.line(0), "A");
  ASSERT_EQ(doc.line(1), "B");
  ASSERT_EQ(doc.line(2), "C");
}

TEST(LINEAR_DOCUMENT, reverse_append)
{
  pp::LinearDocument doc{pp::LinearDocument::Direction::Reverse};

  ASSERT_EQ(doc.lines(), 0);

  doc.append("A");
  doc.append("B");
  doc.append("C");

  ASSERT_EQ(doc.lines(), 3);
  ASSERT_EQ(doc.line(0), "C");
  ASSERT_EQ(doc.line(1), "B");
  ASSERT_EQ(doc.line(2), "A");
}

struct TwoLineDocument final : public pp::MultiLineText
{
  uint32_t lines(void) const override { return 2; }

  const std::string &line(uint32_t n) const override { return _lines[n]; }

  std::string _lines[2];
};

TEST(LINEAR_DOCUMENT, append_multi_line_text)
{
  pp::LinearDocument doc;
  TwoLineDocument sub;

  sub._lines[0] = "B";
  sub._lines[1] = "  C";

  doc.append("A");
  doc.indent();

  doc.append(sub);
  doc.unindent();
  doc.append("D");

  ASSERT_EQ(doc.lines(), 4);

  ASSERT_EQ(doc.line(0), "A");
  ASSERT_EQ(doc.line(1), "  B");
  ASSERT_EQ(doc.line(2), "    C");
  ASSERT_EQ(doc.line(3), "D");
}

TEST(LINEAR_DOCUMENT, document_append)
{
  pp::LinearDocument doc{pp::LinearDocument::Direction::Forward};
  pp::LinearDocument sub{pp::LinearDocument::Direction::Reverse};

  doc.append("A");
  doc.indent();

  sub.append("D");
  sub.indent();
  sub.append("C");
  sub.unindent();
  sub.append("B");

  doc.append(sub);
  doc.unindent();
  doc.append("E");

  ASSERT_EQ(doc.lines(), 5);

  ASSERT_EQ(doc.line(0), "A");
  ASSERT_EQ(doc.line(1), "  B");
  ASSERT_EQ(doc.line(2), "    C");
  ASSERT_EQ(doc.line(3), "  D");
  ASSERT_EQ(doc.line(4), "E");
}

TEST(LINEAR_DOCUMENT, indent)
{
  pp::LinearDocument doc;

  ASSERT_EQ(doc.lines(), 0);

  doc.append("A");
  doc.indent();
  doc.append("B");
  doc.unindent();
  doc.append("C");

  ASSERT_EQ(doc.lines(), 3);

  ASSERT_EQ(doc.line(0), "A");
  ASSERT_EQ(doc.line(1), "  B");
  ASSERT_EQ(doc.line(2), "C");
}
