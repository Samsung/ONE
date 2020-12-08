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

#include "util/EventWriter.h"

#include <sstream>
#include <vector>
#include <unordered_map>
#include <json/json.h>
#include <assert.h>
#include <utility>
#include <map>
#include <set>
#include <stdint.h>
#include <fstream>

// json type for Chrome Event Trace
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
  content.args = evt.args;
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

void ChromeTracingWriter::flush(const std::vector<std::unique_ptr<EventRecorder>> &recorders)
{
  _os << "{\n";
  _os << "  " << quote("traceEvents") << ": [\n";

  for (auto &recorder : recorders)
  {
    flushOneRecord(*recorder);
  }

  _os << "    { }\n";
  _os << "  ]\n";
  _os << "}\n";
}

void ChromeTracingWriter::flushOneRecord(const EventRecorder &recorder)
{
  for (auto &evt : recorder.duration_events())
  {
    _os << "    " << object(evt) << ",\n";
  }

  for (auto &evt : recorder.counter_events())
  {
    _os << "    " << object(evt) << ",\n";
  }
}
