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

#include "Knob.h"

#include <pepper/strcast.h>

#include <iostream>
#include <string>
#include <map>

// Basic Infrastructure to declare and access Knob values
namespace
{

using KnobName = std::string;

/**
 * @brief Load configuration (from somewhere)
 */
struct KnobLoader
{
  virtual ~KnobLoader() = default;

  virtual bool load(const KnobName &name, bool default_value) const = 0;
};

/**
 * @brief Load configuration from environment variables
 *
 * Given a prefix P, EnvKnobLoader reads a configuration K from concat(P, K).
 *
 * For example, let us assume that P is "MY_" and K is "CONFIG".
 *
 * Then, EnvKnobLoader reads configuration CONFIG from environment variable MY_CONFIG.
 */
class EnvKnobLoader final : public KnobLoader
{
public:
  EnvKnobLoader() = default;

public:
  bool load(const KnobName &knob_name, bool default_value) const override
  {
    auto envvar = _prefix + knob_name;
    auto s = std::getenv(envvar.c_str());

    return pepper::safe_strcast<int>(s, default_value ? 1 : 0) != 0;
  }
  void knob_set(const KnobName &knob_name, bool value) { _knob[knob_name] = value; }
  void dialect_set(const exo::Dialect &dialect_name) { _prefix = _label[dialect_name]; }
  bool knob_get(const KnobName &knob_name) { return load(knob_name, _knob[knob_name]); }

private:
  /// @brief Environment variable prefix
  std::string _prefix;
  std::map<KnobName, bool> _knob;
  std::map<exo::Dialect, KnobName> _label = {{exo::Dialect::TFLITE, "TFL_"},
                                             {exo::Dialect::CIRCLE, "CIRCLE_"}};
};

} // namespace

namespace
{

EnvKnobLoader &knob_loader(void)
{
  // TODO separate "EXOTFLITE_" and "EXOCIRCLE_" when necessary
  static EnvKnobLoader loader;
  return loader;
}

} // namespace

namespace exo
{

#define KNOB_BOOL(NAME, TFL_DEFAULT, CIRCLE_DEFAULT, DESC)                    \
  template <> typename KnobTrait<Knob::NAME>::ValueType get<Knob::NAME>(void) \
  {                                                                           \
    return ::knob_loader().knob_get(#NAME);                                   \
  }
#include "Knob.lst"
#undef KNOB_BOOL

void set(Dialect d)
{
  ::knob_loader().dialect_set(d);
  switch (d)
  {
    case Dialect::TFLITE:
#define KNOB_BOOL(NAME, TFL_DEFAULT, CIRCLE_DEFAULT, DESC) \
  ::knob_loader().knob_set(#NAME, TFL_DEFAULT);
#include "Knob.lst"
#undef KNOB_BOOL
      break;
    case Dialect::CIRCLE:
#define KNOB_BOOL(NAME, TFL_DEFAULT, CIRCLE_DEFAULT, DESC) \
  ::knob_loader().knob_set(#NAME, CIRCLE_DEFAULT);
#include "Knob.lst"
#undef KNOB_BOOL
      break;
    default:
      std::runtime_error("UnKnown dialect");
  }
}

} // namespace exo
