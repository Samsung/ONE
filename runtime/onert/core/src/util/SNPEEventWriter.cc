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

#include <json/json.h>

#include <cassert>
#include <unordered_map>
#include <utility>

/**
 * @brief Version of SNPE format
 * In version 1
 * - There is no "version" field in Json
 * - Only one subgraph is supported
 * - Operation name is a form of "$3 ADD"
 *
 * In version 2,
 * - "version" : "2" was added in Json
 * - Multiple session and multiple subgraphs are supported
 * - When there is only one session, operation name is a form of "$2 subgraph $3 ADD",
 *   meaning ADD op whose operation index 3 in a subgraph whose index is 2
 * - When there are two or more sessions, operation name is a form of
 *   "$1 session $2 subgraph $3 ADD", meaning ADD op whose operation index 3
 *   in a subgraph whose index is 2, which was run in 1st session.
 */
#define SNPE_JSON_SCHEMA_VERSION "2"

namespace
{

std::string getLabel(const DurationEvent &evt)
{
  if (auto evt_ptr = dynamic_cast<const OpSeqDurationEvent *>(&evt))
  {
    std::string subg_label("$" + std::to_string(evt_ptr->subg_index) + " subgraph");
    std::string op_label("$" + std::to_string(evt_ptr->op_index) + " " + evt_ptr->op_name);

    // Note : At this moment, there is only one thread running for EventWriter
    if (evt_ptr->tracing_ctx->hasMultipleSessions())
    {
      std::string session_label("$" + std::to_string(evt_ptr->session_index) + " session");
      return session_label + " " + subg_label + " " + op_label;
    }
    else
    {
      // When there is only one session, do not include session info
      // Refer to https://github.sec.samsung.net/STAR/nnfw/issues/11436#issuecomment-930332
      return subg_label + " " + op_label;
    }
  }
  else // SubgEvent
    return "Graph";
}

std::string getBackend(const DurationEvent &evt)
{
  if (auto evt_ptr = dynamic_cast<const OpSeqDurationEvent *>(&evt))
    return evt_ptr->backend;
  else // SubbEvent
    return "runtime";
}

} // namespace

void SNPEWriter::flush(const std::vector<std::unique_ptr<EventRecorder>> &recorders)
{
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

  Json::Value root;
  root["version"] = SNPE_JSON_SCHEMA_VERSION;

  auto &exec_data = root["Execution_Data"] = Json::Value{Json::objectValue};

  // Memory
  {
    std::unordered_map<std::string, Stat> mem_stats;
    for (const auto &recorder : recorders)
    {
      for (const auto &evt : recorder->counter_events())
      {
        auto &mem_stat = mem_stats[evt.name];
        uint64_t val = std::stoull(evt.values.at("value"));
        mem_stat.accumulate(val);
      }
    }

    auto &mem = exec_data["memory"] = Json::Value{Json::objectValue};
    for (const auto &[key, val] : mem_stats)
    {
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
    for (const auto &recorder : recorders)
    {
      for (const auto &evt : recorder->duration_events())
      {
        std::string evt_name = getLabel(*evt);
        std::string evt_tid = getBackend(*evt);

        auto &stat = stats[evt_tid][evt_name];
        auto &begin_ts = begin_timestamps[evt_tid][evt_name];
        uint64_t timestamp = std::stoull(evt->ts);
        if (evt->ph == "B")
        {
          if (begin_ts != 0)
            throw std::runtime_error{"Invalid Data"};
          begin_ts = timestamp;
        }
        else if (evt->ph == "E")
        {
          if (begin_ts == 0 || timestamp < begin_ts)
            throw std::runtime_error{"Invalid Data"};
          stat.accumulate(timestamp - begin_ts);
          begin_ts = 0;
        }
        else
          throw std::runtime_error{"Invalid Data - invalid value for \"ph\" : \"" + evt->ph + "\""};
      }
    }

    for (const auto &kv : begin_timestamps)
      for (const auto &kv2 : kv.second)
        if (kv2.second != 0)
          throw std::runtime_error{"Invalid Data - B and E pair does not match."};

    for (const auto &[tid, stat_map] : stats)
    {
      auto &json_tid = exec_data[tid] = Json::Value{Json::objectValue};
      for (const auto &[name, val] : stat_map)
      {
        json_tid[name]["Avg_Time"] = val.sum / val.count;
        json_tid[name]["Max_Time"] = val.max;
        json_tid[name]["Min_Time"] = val.min;
        json_tid[name]["Runtime"] = tid;
      }
    }
  }

  _os << root;
}
