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

#include <stdexcept>

namespace luci
{

class UserSettingsImpl : public UserSettings
{
public:
  void set(const Key key, bool value) override;
  bool get(const Key key) const override;

private:
  bool _MuteWarnings{false};
  bool _DisableValidation{false};
  bool _ProfilingDataGen{false};
  bool _ExecutionPlanGen{false};
};

void UserSettingsImpl::set(const Key key, bool value)
{
  switch (key)
  {
    case Key::MuteWarnings:
      _MuteWarnings = value;
      break;
    case Key::DisableValidation:
      _DisableValidation = value;
      break;
    case Key::ProfilingDataGen:
      _ProfilingDataGen = value;
      break;
    case Key::ExecutionPlanGen:
      _ExecutionPlanGen = value;
      break;
    default:
      throw std::runtime_error("Invalid key in boolean set");
      break;
  }
}

bool UserSettingsImpl::get(const Key key) const
{
  switch (key)
  {
    case Key::MuteWarnings:
      return _MuteWarnings;
    case Key::DisableValidation:
      return _DisableValidation;
    case Key::ProfilingDataGen:
      return _ProfilingDataGen;
    case Key::ExecutionPlanGen:
      return _ExecutionPlanGen;
    default:
      throw std::runtime_error("Invalid key in boolean get");
      break;
  }
  return false;
}

} // namespace luci

namespace luci
{

UserSettings *UserSettings::settings()
{
  static UserSettingsImpl _this;
  return &_this;
}

} // namespace luci
