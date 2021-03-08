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

#ifndef __LUCI_USER_SETTINGS__
#define __LUCI_USER_SETTINGS__

// NOTE Revise the logic if we find a better way not using global status

namespace luci
{

/**
 * @brief UserSettings provides user settings by key-value
 */
struct UserSettings
{
  enum Key
  {
    Undefined,
    MuteWarnings,
    DisableValidation,
    ProfilingDataGen,
  };

  static UserSettings *settings();

  virtual void set(const Key key, bool value) = 0;
  virtual bool get(const Key key) const = 0;
};

} // namespace luci

#endif // __LUCI_USER_SETTINGS__
