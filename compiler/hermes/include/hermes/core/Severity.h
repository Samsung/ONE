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

#ifndef __HERMES_SEVERITY_H__
#define __HERMES_SEVERITY_H__

#include <cstdint>

namespace hermes
{

/**
 * FATAL > ERROR > WARN > INFO > VERBOSE
 *
 * Hermes deliberately declares SeverityCategory as "enum" (instead of "enum class")
 * in order to reduce namespace nesting.
 */
enum SeverityCategory : uint16_t
{
  FATAL = 0,
  ERROR = 1,
  WARN = 2,
  INFO = 3,
  VERBOSE = 4,
};

class Severity final
{
public:
  friend Severity fatal(void);
  friend Severity error(void);
  friend Severity warn(void);
  friend Severity info(void);
  friend Severity verbose(uint16_t level);

private:
  /**
   * Use below "factory" helpers.
   */
  Severity(SeverityCategory cat, uint16_t lvl) : _cat{cat}, _lvl{lvl}
  {
    // DO NOTHING
  }

public:
  const SeverityCategory &category(void) const { return _cat; }

  /**
   * @brief Verbose level
   *
   * "level" is fixed as 0 for all the categories except VERBOSE.
   *
   * 0 (most significant) <--- level ---> 65535 (least significant)
   */
  const uint16_t &level(void) const { return _lvl; }

private:
  SeverityCategory _cat;
  uint16_t _lvl;
};

inline Severity fatal(void) { return Severity{FATAL, 0}; }
inline Severity error(void) { return Severity{ERROR, 0}; }
inline Severity warn(void) { return Severity{WARN, 0}; }
inline Severity info(void) { return Severity{INFO, 0}; }
inline Severity verbose(uint16_t level) { return Severity{VERBOSE, level}; }

} // namespace hermes

#endif // __HERMES_SEVERITY_H__
