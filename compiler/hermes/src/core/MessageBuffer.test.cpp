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

#include "hermes/core/MessageBuffer.h"

#include <cassert>

#include <gtest/gtest.h>

namespace
{

class MockMessageBus final : public hermes::MessageBus
{
public:
  MockMessageBus() = default;

public:
  void post(std::unique_ptr<hermes::Message> &&msg) override
  {
    _count += 1;
    _message = std::move(msg);
  }

public:
  uint32_t count(void) const { return _count; }
  const hermes::Message *message(void) const { return _message.get(); }

private:
  unsigned _count = 0;
  std::unique_ptr<hermes::Message> _message = nullptr;
};

} // namespace

TEST(MessageBufferTest, pass_constructed_message_on_descturction)
{
  MockMessageBus bus;

  {
    hermes::MessageBuffer buf{&bus};

    buf.os() << "Hello" << std::endl;
    buf.os() << "Nice to meet you" << std::endl;
  }

  ASSERT_EQ(bus.count(), 1);
  ASSERT_NE(bus.message(), nullptr);
  ASSERT_NE(bus.message()->text(), nullptr);
  ASSERT_EQ(bus.message()->text()->lines(), 2);
  ASSERT_EQ(bus.message()->text()->line(0), "Hello");
  ASSERT_EQ(bus.message()->text()->line(1), "Nice to meet you");
}
