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

#include "hermes/ConsoleReporter.h"

#include <memory>
#include <sstream>

#include <gtest/gtest.h>

TEST(ConsoleReporterTest, constructor)
{
  hermes::ConsoleReporter r;

  SUCCEED();
}

TEST(ConsoleReporterTest, notify)
{
  hermes::Message m;
  {
    std::stringstream ss;

    ss << "Hello" << std::endl;

    m.text(std::make_unique<hermes::MessageText>(ss));
  }

  hermes::ConsoleReporter r;

  ASSERT_NO_THROW(r.notify(&m));
}
