/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "EqualizePatternRead.h"

#include <fstream>
#include <json.h>

using namespace fme_apply;

namespace
{

EqualizePattern::Type eq_type(const std::string &type)
{
  if (type == "ScaleOnly")
    return EqualizePattern::ScaleOnly;
  if (type == "ShiftOnly")
    return EqualizePattern::ShiftOnly;
  if (type == "ScaleShift")
    return EqualizePattern::ScaleShift;

  throw std::runtime_error("Unsupported equalization pattern");
}

} // namespace

namespace fme_apply
{

std::vector<EqualizePattern> read(const std::string &filename)
{
  Json::Value root;
  std::ifstream ifs(filename);

  // Failed to open cfg file
  if (not ifs.is_open())
    throw std::runtime_error("Cannot open config file. " + filename);

  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;

  // Failed to parse
  if (not parseFromStream(builder, ifs, &root, &errs))
    throw std::runtime_error("Cannot parse config file (json format). " + errs);

  std::vector<EqualizePattern> res;

  for (auto &eq_pattern : root)
  {
    auto get_string = [&](const std::string &val) {
      if (not eq_pattern.isMember(val))
        throw std::runtime_error(val + " is missing in " + filename);
      if (not eq_pattern[val].isString())
        throw std::runtime_error(val + " is not string");

      return eq_pattern[val].asString();
    };

    auto get_fp32_array = [&](const std::string &val) {
      if (not eq_pattern.isMember(val))
        throw std::runtime_error(val + " is missing in " + filename);
      auto arr = eq_pattern[val];
      if (not arr.isArray())
        throw std::runtime_error(val + " is not array");

      std::vector<float> res;
      for (auto &elem : arr)
      {
        if (not elem.isNumeric())
          throw std::runtime_error(val + "'s element is not fp32");

        res.emplace_back(elem.asFloat());
      }

      return res;
    };

    auto front = get_string("front");
    auto back = get_string("back");
    auto type = get_string("type");

    EqualizePattern p;
    {
      p.front = front;
      p.back = back;
      p.type = eq_type(type);
      switch (p.type)
      {
        case EqualizePattern::Type::ScaleOnly:
          p.act_scale = get_fp32_array("act_scale");
          break;
        default:
          throw std::runtime_error("Unsupported EqualizePattern type");
      }
    }
    res.emplace_back(p);
  }

  return res;
}

} // namespace fme_apply
