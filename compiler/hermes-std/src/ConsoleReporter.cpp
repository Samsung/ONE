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

#include <iostream>
#include <cstdlib>
#include <string>

namespace hermes
{

static constexpr const char *kTermColorRedTextCode = "\033[0;31m";
static constexpr const char *kTermColorGreenTextCode = "\033[0;32m";
static constexpr const char *kTermColorOrangeTextCode = "\033[0;33m";
static constexpr const char *kTermColorBlueTextCode = "\033[0;34m";
static constexpr const char *kTermColorMagentaTextCode = "\033[0;35m";
static constexpr const char *kTermColorCyanTextCode = "\033[0;36m";
static constexpr const char *kTermColorWhiteTextCode = "\033[0;37m";

static constexpr const char *kTermBoldTextCode = "\033[1m";
static constexpr const char *kTermUnderlineTextCode = "\033[4m";
static constexpr const char *kTermInverseTextCode = "\033[7m";
static constexpr const char *kTermBoldOffTextCode = "\033[21m";
static constexpr const char *kTermUnderlineOffTextCode = "\033[24m";
static constexpr const char *kTermInverseOffTextCode = "\033[27m";

static constexpr const char *kTermColorResetAllCode = "\033[0m";

void ConsoleReporter::notify(const hermes::Message *m)
{
  if (not _env_checked)
  {
    const char *env_color_p = std::getenv("ONE_HERMES_COLOR");
    if (env_color_p)
    {
      auto env_color_str = std::string(env_color_p);
      if ((env_color_str == "1") or (env_color_str == "ON"))
        _is_colored = true;
    }
    _env_checked = true;
  }

  if (_is_colored)
  {
    switch (m->get_severity())
    {
      case FATAL:
        std::cout << kTermColorRedTextCode << kTermBoldTextCode << kTermUnderlineTextCode;
        break;
      case ERROR:
        std::cout << kTermColorRedTextCode;
        break;
      case WARN:
        std::cout << kTermColorOrangeTextCode;
        break;
      case INFO:
        std::cout << kTermColorGreenTextCode;
        break;
      case VERBOSE:
        std::cout << kTermColorResetAllCode;
        break;
    };
  }
  for (uint32_t n = 0; n < m->text()->lines(); ++n)
  {
    std::cout << m->text()->line(n) << std::endl;
  }
  if (_is_colored)
  {
    std::cout << kTermColorResetAllCode;
  }
}

} // namespace hermes
