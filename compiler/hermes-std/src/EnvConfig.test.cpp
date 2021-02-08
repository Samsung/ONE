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

#include "hermes/EnvConfig.h"

#include <hermes/core/SourceSetting.h>

#include <gtest/gtest.h>

#include <stdlib.h>

namespace
{

class Logger final : public hermes::Source
{
public:
  Logger() = default;
  ~Logger() = default;
};

std::string env_name("TEST_CONFIG");

} // namespace

TEST(EnvConfigTest, constructor)
{
  hermes::EnvConfig<hermes::EnvFormat::BooleanNumber> ec(env_name);

  SUCCEED();
}

TEST(EnvConfigTest, configure)
{
  Logger logger;
  hermes::SourceSetting ss;
  hermes::EnvConfig<hermes::EnvFormat::BooleanNumber> ec(env_name);

  ec.configure(&logger, ss);

  SUCCEED();
}

TEST(EnvConfigTest, configure_enabled)
{
  setenv(env_name.c_str(), "1", 0);

  Logger logger;
  hermes::SourceSetting ss;
  hermes::EnvConfig<hermes::EnvFormat::BooleanNumber> ec(env_name);

  ec.configure(&logger, ss);

  SUCCEED();
}
