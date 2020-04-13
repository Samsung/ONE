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

#include "hermes/core/Source.h"

#include <gtest/gtest.h>

namespace
{

struct MockSourceRegistry final : public hermes::Source::Registry
{
  void attach(hermes::Source *) override { return; }
  void detach(hermes::Source *) override { return; }
};

struct MockMessageBus final : public hermes::MessageBus
{
  void post(std::unique_ptr<hermes::Message> &&msg) override
  {
    msg.reset();
    ++cnt;
  }

  uint32_t cnt = 0;
};

struct MockSource final : public hermes::Source
{
  MockSource(hermes::Source::Registry *r, hermes::MessageBus *b) { activate(r, b); }
  ~MockSource() { deactivate(); }

  void reload(const hermes::Config *) override { return; }

  void enable(void) { setting().accept_all(); }
};

} // namespace

TEST(SourceTest, construct)
{
  MockSourceRegistry registry;
  MockMessageBus bus;

  MockSource source{&registry, &bus};

  // Source are off at the beginning
  ASSERT_FALSE(source.check(::hermes::fatal()));
  ASSERT_FALSE(source.check(::hermes::error()));
  ASSERT_FALSE(source.check(::hermes::warn()));
  ASSERT_FALSE(source.check(::hermes::info()));
  ASSERT_FALSE(source.check(::hermes::verbose(100)));
}

TEST(SourceTest, macro)
{
  MockSourceRegistry registry;

  MockMessageBus bus;

  MockSource source{&registry, &bus};

  source.enable();

  uint32_t expected_count = 0;

  // No message at the beginning
  ASSERT_EQ(bus.cnt, 0);

  HERMES_ERROR(source) << "A";
  ASSERT_EQ(bus.cnt, ++expected_count);

  HERMES_WARN(source) << "A";
  ASSERT_EQ(bus.cnt, ++expected_count);

  HERMES_INFO(source) << "A";
  ASSERT_EQ(bus.cnt, ++expected_count);

  HERMES_VERBOSE(source, 100) << "A";
  ASSERT_EQ(bus.cnt, ++expected_count);

// FATAL message should terminate the execution. Let's check how to check this!
// TODO Enable FATAL feature and enable this test
#if 0
  HERMES_FATAL(source) << "A";
  ASSERT_EQ(bus.cnt, 1);
#endif
}
