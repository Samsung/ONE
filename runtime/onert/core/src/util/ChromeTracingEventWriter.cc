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

#include "EventWriter.h"

#include <cassert>
#include <sstream>
#include <utility>
#include <vector>

// json type for ChromeTracingWriter
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

void fill(Content &content, const DurationEvent &evt, const std::string &name,
          const std::string &tid)
{
  content.flds.emplace_back("name", name);
  content.flds.emplace_back("pid", "0");
  content.flds.emplace_back("tid", tid);
  content.flds.emplace_back("ph", evt.ph);
  content.flds.emplace_back("ts", evt.ts);
  content.args = evt.args;
}

void fill(Content &content, const CounterEvent &evt)
{
  assert(evt.name != "");

  content.flds.emplace_back("name", evt.name);
  content.flds.emplace_back("pid", "0");
  content.flds.emplace_back("tid", evt.tid);
  content.flds.emplace_back("ph", evt.ph);
  content.flds.emplace_back("ts", evt.ts);
  content.args = evt.args;
}

std::string object(const DurationEvent &evt, const std::string &name, const std::string &tid)
{
  Content content;

  fill(content, evt, name, tid);

  return ::object(content);
}

std::string object(const CounterEvent &evt)
{
  Content content;

  fill(content, evt);

  for (const auto &[key, val] : evt.values)
  {
    content.args.emplace_back(key, val);
  }

  return ::object(content);
}

std::string getSessionLabel(const DurationEvent &evt)
{
  return "$" + std::to_string(evt.session_index) + " sess";
}

std::string getSubgLabel(const DurationEvent &evt)
{
  return "$" + std::to_string(evt.subg_index) + " subg";
}

std::string getOpLabel(const OpSeqDurationEvent &evt)
{
  return "@" + std::to_string(evt.op_index) + " " + evt.op_name;
}

std::string getLabel(const DurationEvent &evt)
{
  if (auto evt_ptr = dynamic_cast<const OpSeqDurationEvent *>(&evt))
  {
    return getOpLabel(*evt_ptr);
  }
  else // SubgDurationEvent
  {
    return getSubgLabel(evt);
  }
}

std::string getTid(const DurationEvent &evt)
{
  if (auto evt_ptr = dynamic_cast<const OpSeqDurationEvent *>(&evt))
  {
    return getSessionLabel(*evt_ptr) + ", " + getSubgLabel(*evt_ptr) + ", " + evt_ptr->backend;
  }
  else // SubgDurationEvent
  {
    return getSessionLabel(evt) + ", " + getSubgLabel(evt);
  }
}

} // namespace

ChromeTracingWriter::ChromeTracingWriter(const std::string &filepath) : EventFormatWriter(filepath)
{
  _os << "{\n";
  _os << "  " << quote("traceEvents") << ": [\n";
}

ChromeTracingWriter::~ChromeTracingWriter()
{
  _os << "    { }\n";
  _os << "  ]\n";
  _os << "}\n";
}

void ChromeTracingWriter::flush(const std::vector<std::unique_ptr<EventRecorder>> &recorders)
{
  for (const auto &recorder : recorders)
  {
    flushOneRecord(*recorder);
  }
}

void ChromeTracingWriter::flushOneRecord(const EventRecorder &recorder)
{
  for (const auto &evt : recorder.duration_events())
  {
    const std::string name = getLabel(*evt);
    const std::string tid = getTid(*evt);

    _os << "    " << object(*evt, name, tid) << ",\n";
  }

  for (const auto &evt : recorder.counter_events())
  {
    _os << "    " << object(evt) << ",\n";
  }
}
