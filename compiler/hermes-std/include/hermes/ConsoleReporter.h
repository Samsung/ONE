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

#ifndef __HERMES_STD_CONSOLE_REPORTER_H__
#define __HERMES_STD_CONSOLE_REPORTER_H__

#include <hermes.h>

namespace hermes
{

/**
 * @brief Print messages into standard console
 */
struct ConsoleReporter final : public hermes::Sink
{
  void notify(const Message *m) final;
  void set_colored_mode(bool is_colored) { _is_colored = is_colored; }

private:
  bool _is_colored = false;
  bool _env_checked = false;
};

} // namespace hermes

#endif // __HERMES_STD_CONSOLE_REPORTER_H__
