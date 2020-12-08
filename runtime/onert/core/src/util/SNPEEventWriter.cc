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

#include <unordered_map>
#include <json/json.h>
#include <cassert>
#include <utility>

void SNPEWriter::flush(const std::vector<std::unique_ptr<EventRecorder>> &recorders)
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
    for (auto &recorder : recorders)
    {
      for (auto &evt : recorder->counter_events())
      {
        auto &mem_stat = mem_stats[evt.name];
        uint64_t val = std::stoull(evt.values.at("value"));
        mem_stat.accumulate(val);
      }
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
    for (auto &recorder : recorders)
    {
      for (auto &evt : recorder->duration_events())
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

  _os << root;
}
