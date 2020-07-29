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

#include "hermes.h"

#include <gtest/gtest.h>

namespace
{

class Logger final : public hermes::Source
{
public:
  Logger(hermes::Context *ctx);
  ~Logger();
};

Logger::Logger(hermes::Context *ctx) { activate(ctx->sources(), ctx->bus()); }
Logger::~Logger() { deactivate(); }

} // namespace

TEST(HermesTest, logger_constructor_NEG)
{
  hermes::Context context;
  // we expect segmentfault from nullptr->sources()
  ASSERT_DEATH(Logger logger(&context), "");

  SUCCEED();
}

// TODO add HermesTest simple_usecase
