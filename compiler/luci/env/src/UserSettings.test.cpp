/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/UserSettings.h"

#include <gtest/gtest.h>

TEST(UserSettings, instance)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  auto s2 = luci::UserSettings::settings();
  ASSERT_EQ(s2, settings);
}

TEST(UserSettings, MuteWarnings)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  settings->set(luci::UserSettings::Key::MuteWarnings, false);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::MuteWarnings));

  settings->set(luci::UserSettings::Key::MuteWarnings, true);
  ASSERT_TRUE(settings->get(luci::UserSettings::Key::MuteWarnings));
}

TEST(UserSettings, MuteWarnings_NEG)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  settings->set(luci::UserSettings::Key::MuteWarnings, false);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::MuteWarnings));

  settings->set(luci::UserSettings::Key::MuteWarnings, true);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::DisableValidation));
}

TEST(UserSettings, DisableValidation)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  settings->set(luci::UserSettings::Key::DisableValidation, false);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::DisableValidation));

  settings->set(luci::UserSettings::Key::DisableValidation, true);
  ASSERT_TRUE(settings->get(luci::UserSettings::Key::DisableValidation));
}

TEST(UserSettings, DisableValidation_NEG)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  settings->set(luci::UserSettings::Key::DisableValidation, false);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::DisableValidation));

  settings->set(luci::UserSettings::Key::DisableValidation, true);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::ProfilingDataGen));
}

TEST(UserSettings, ProfilingDataGen)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  settings->set(luci::UserSettings::Key::ProfilingDataGen, false);
  ASSERT_FALSE(settings->get(luci::UserSettings::Key::ProfilingDataGen));

  settings->set(luci::UserSettings::Key::ProfilingDataGen, true);
  ASSERT_TRUE(settings->get(luci::UserSettings::Key::ProfilingDataGen));
}

TEST(UserSettings, undefined_set_NEG)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  ASSERT_THROW(settings->set(luci::UserSettings::Key::Undefined, true), std::exception);
}

TEST(UserSettings, undefined_get_NEG)
{
  auto settings = luci::UserSettings::settings();
  ASSERT_NE(nullptr, settings);

  ASSERT_THROW(settings->get(luci::UserSettings::Key::Undefined), std::exception);
}
