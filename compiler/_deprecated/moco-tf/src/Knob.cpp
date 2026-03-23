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

// Basic Infrastructure to declare and access Knob values
//
// TODO Reuse this infrastructure as a library
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

// Template-programming helpers
template <typename T> T knob_load(const KnobLoader &, const KnobName &, const T &);

template <>
bool knob_load(const KnobLoader &l, const KnobName &knob_name, const bool &default_value)
{
  return l.load(knob_name, default_value);
}

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
  EnvKnobLoader(const std::string &prefix) : _prefix{prefix}
  {
    // DO NOTHING
  }

public:
  bool load(const KnobName &knob_name, bool default_value) const override
  {
    auto envvar = _prefix + knob_name;
    auto s = std::getenv(envvar.c_str());

    return pepper::safe_strcast<int>(s, default_value ? 1 : 0) != 0;
  }

private:
  /// @brief Environment variable prefix
  std::string _prefix;
};

} // namespace

namespace
{

/**
 * TODO Support Knob Loader Injection
 *
 * Let us assume that there is a compiler "A" based on moco, and it wants to reuse this
 * infrastructure.
 *
 * Under the current design, users have to set "MOCO_XXX" even though they uses "A", which is
 * counter-intuitive.
 *
 * "Knob Loader Injection" aims to address this issue. "Knob Loader Injection" allows "A" to
 * inject its own knob loader that reads "A_XXX" environment variables.
 */
const KnobLoader &knob_loader(void)
{
  static EnvKnobLoader loader{"MOCO_"};
  return loader;
}

} // namespace

namespace moco
{
namespace tf
{

#define KNOB_BOOL(NAME, DEFAULT, DESC)                                                         \
  template <> typename KnobTrait<Knob::NAME>::ValueType get<Knob::NAME>(void)                  \
  {                                                                                            \
    static typename KnobTrait<Knob::NAME>::ValueType value =                                   \
      ::knob_load<typename KnobTrait<Knob::NAME>::ValueType>(::knob_loader(), #NAME, DEFAULT); \
    return value;                                                                              \
  }
#include "Knob.lst"
#undef KNOB_BOOL

} // namespace tf
} // namespace moco
