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

#include "util/EventRecorder.h"

#include <sstream>
#include <vector>
#include <unordered_map>
#include <json/json.h>
#include <assert.h>

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

  _duration_events.push_back(evt);
}

void EventRecorder::emit(const CounterEvent &evt)
{
  std::lock_guard<std::mutex> lock{_mu};

  _counter_events.push_back(evt);
}

void EventRecorder::writeToFile(std::ostream &os)
{
  std::lock_guard<std::mutex> lock{_mu};

  switch (_write_format)
  {
    case WriteFormat::CHROME_TRACING:
      writeChromeTrace(os);
      break;
    case WriteFormat::SNPE_BENCHMARK:
      writeSNPEBenchmark(os);
      break;
    default:
      assert(!"Invalid value");
      break;
  }
}

void EventRecorder::writeSNPEBenchmark(std::ostream &os)
{
  Json::Value root;
  auto &exec_data = root["Execution_Data"] = Json::Value{Json::objectValue};

  struct Stat
  {
    uint64_t sum = 0;
    uint64_t count = 0;
    uint64_t max = 0;
    uint64_t min = std::numeric_limits<uint64_t>::max();

    void accumulate(uint64_t val)
    {
      sum += val;
      count++;
      max = std::max(max, val);
      min = std::min(min, val);
    }
  };

  // Memory
  {
    std::unordered_map<std::string, Stat> mem_stats;
    for (auto &evt : _counter_events)
    {
      auto &mem_stat = mem_stats[evt.name];
      uint64_t val = std::stoull(evt.values["value"]);
      mem_stat.accumulate(val);
    }

    auto &mem = exec_data["memory"] = Json::Value{Json::objectValue};
    for (auto &kv : mem_stats)
    {
      auto &key = kv.first;
      auto &val = kv.second;
      mem[key]["Avg_Size"] = val.sum / val.count;
      mem[key]["Max_Size"] = val.max;
      mem[key]["Min_Size"] = val.min;
      mem[key]["Runtime"] = "NA";
    }
  }

  // Operation Execution Time
  {
    // NOTE This assumes _duration_events is sorted by "ts" ascending

    // 2D keys : stats[tid][name]
    std::unordered_map<std::string, std::unordered_map<std::string, Stat>> stats;
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> begin_timestamps;
    for (auto &evt : _duration_events)
    {
      auto &stat = stats[evt.tid][evt.name];
      auto &begin_ts = begin_timestamps[evt.tid][evt.name];
      uint64_t timestamp = std::stoull(evt.ts);
      if (evt.ph == "B")
      {
        if (begin_ts != 0)
          throw std::runtime_error{"Invalid Data"};
        begin_ts = timestamp;
      }
      else if (evt.ph == "E")
      {
        if (begin_ts == 0 || timestamp < begin_ts)
          throw std::runtime_error{"Invalid Data"};
        stat.accumulate(timestamp - begin_ts);
        begin_ts = 0;
      }
      else
        throw std::runtime_error{"Invalid Data - invalid value for \"ph\" : \"" + evt.ph + "\""};
    }

    for (auto &kv : begin_timestamps)
      for (auto &kv2 : kv.second)
        if (kv2.second != 0)
          throw std::runtime_error{"Invalid Data - B and E pair does not match."};

    for (auto &kv : stats)
    {
      auto &tid = kv.first;
      auto &map = kv.second;
      auto &json_tid = exec_data[tid] = Json::Value{Json::objectValue};
      for (auto &kv : map)
      {
        auto &name = kv.first;
        auto &val = kv.second;
        json_tid[name]["Avg_Time"] = val.sum / val.count;
        json_tid[name]["Max_Time"] = val.max;
        json_tid[name]["Min_Time"] = val.min;
        json_tid[name]["Runtime"] = tid;
      }
    }
  }

  os << root;
}

void EventRecorder::writeChromeTrace(std::ostream &os)
{
  os << "{\n";
  os << "  " << quote("traceEvents") << ": [\n";

  for (auto &evt : _duration_events)
  {
    os << "    " << object(evt) << ",\n";
  }

  for (auto &evt : _counter_events)
  {
    os << "    " << object(evt) << ",\n";
  }

  os << "    { }\n";
  os << "  ]\n";
  os << "}\n";
}
