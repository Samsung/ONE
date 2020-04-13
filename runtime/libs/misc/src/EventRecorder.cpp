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

#include "misc/EventRecorder.h"

#include <sstream>
#include <vector>

namespace
{

std::string quote(const std::string &value)
{
  std::stringstream ss;
  ss << '"' << value << '"';
  return ss.str();
}

std::string field(const std::string &k, const std::string &v)
{
  std::stringstream ss;
  ss << quote(k) << " : " << quote(v);
  return ss.str();
}

struct Content // One Entry in Chrome Event Trace
{
  std::vector<std::pair<std::string, std::string>> flds;
  std::vector<std::pair<std::string, std::string>> args;
};

std::string object(const Content &content)
{
  std::stringstream ss;

  ss << "{ ";

  ss << field(content.flds[0].first, content.flds[0].second);

  for (uint32_t n = 1; n < content.flds.size(); ++n)
  {
    ss << ", " << field(content.flds.at(n).first, content.flds.at(n).second);
  }

  if (content.args.size() > 0)
  {
    ss << ", " << quote("args") << " : { ";
    ss << field(content.args.at(0).first, content.args.at(0).second);

    for (uint32_t n = 1; n < content.args.size(); ++n)
    {
      ss << ", " << field(content.args.at(n).first, content.args.at(n).second);
    }

    ss << "}";
  }

  ss << " }";

  return ss.str();
}

void fill(Content &content, const Event &evt)
{
  content.flds.emplace_back("name", evt.name);
  content.flds.emplace_back("pid", "0");
  content.flds.emplace_back("tid", evt.tid);
  content.flds.emplace_back("ph", evt.ph);
  content.flds.emplace_back("ts", evt.ts);
}

std::string object(const DurationEvent &evt)
{
  Content content;

  fill(content, evt);

  return ::object(content);
}

std::string object(const CounterEvent &evt)
{
  Content content;

  fill(content, evt);

  for (auto it = evt.values.begin(); it != evt.values.end(); ++it)
  {
    content.args.emplace_back(it->first, it->second);
  }

  return ::object(content);
}

} // namespace

void EventRecorder::emit(const DurationEvent &evt)
{
  std::lock_guard<std::mutex> lock{_mu};

  _ss << "    " << object(evt) << ",\n";
}

void EventRecorder::emit(const CounterEvent &evt)
{
  std::lock_guard<std::mutex> lock{_mu};

  _ss << "    " << object(evt) << ",\n";
}

void EventRecorder::writeToFile(std::ostream &os)
{
  std::lock_guard<std::mutex> lock{_mu};

  os << "{\n";
  os << "  " << quote("traceEvents") << ": [\n";

  os << _ss.str();

  os << "    { }\n";
  os << "  ]\n";
  os << "}\n";
}
