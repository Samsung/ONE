/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file EnvVar.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::EnvVar class
 */

#ifndef __NNFW_MISC_ENV_VAR__
#define __NNFW_MISC_ENV_VAR__

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

namespace nnfw
{
namespace misc
{
/**
 * @brief Class to access environment variable
 */
class EnvVar
{
public:
  /**
   * @brief Construct a new EnvVar object
   * @param[in] key   environment variable
   */
  EnvVar(const std::string &key)
  {
    const char *value = std::getenv(key.c_str());
    if (value == nullptr)
    {
      // An empty string is considered as an empty value
      _value = "";
    }
    else
    {
      _value = value;
    }
  }

  /**
   * @brief Get environment variable of string type
   * @param[in] def   Default value of environment variable
   * @return Defaut value passed as a parameter when there is no environment variable,
   *         otherwise the value of environment variable passed into constructor
   */
  std::string asString(const std::string &def) const
  {
    if (_value.empty())
      return def;
    return _value;
  }

  /**
   * @brief Get environment variable of boolean type
   * @param[in] def   Default value of environment variable
   * @return Defaut value passed as a parameter when there is no environment variable,
   *         otherwise the value of environment variable passed into constructor
   */
  bool asBool(bool def) const
  {
    if (_value.empty())
      return def;
    static const std::array<std::string, 5> false_list{"0", "OFF", "FALSE", "N", "NO"};
    auto false_found = std::find(false_list.begin(), false_list.end(), _value);
    return (false_found == false_list.end());
  }

  /**
   * @brief Get environment variable of int type
   * @param[in] def   Default value of environment variable
   * @return Defaut value passed as a parameter when there is no environment variable,
   *         otherwise the value of environment variable passed into constructor
   */
  int asInt(int def) const
  {
    if (_value.empty())
      return def;
    return std::stoi(_value);
  }

  /**
   * @brief Get environment variable of float type
   * @param[in] def   Default value of environment variable
   * @return Defaut value passed as a parameter when there is no environment variable,
   *         otherwise the value of environment variable passed into constructor
   */
  float asFloat(float def) const
  {
    if (_value.empty())
      return def;
    return std::stof(_value);
  }

private:
  std::string _value;
};

} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_ENV_VAR__
