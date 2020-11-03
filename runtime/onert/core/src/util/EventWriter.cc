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

// md table type
namespace
{

void writeMDTableRow(std::ostream &os, const std::vector<std::string> &list)
{
  os << "| ";
  for (auto &key : list)
  {
    os << key << " | ";
  }
  os << "\n";
}

struct MDContent
{
  std::string name;
  uint64_t begin_ts;
  uint64_t end_ts;
  uint32_t min_rss;
  uint32_t max_rss;
  uint32_t min_page_reclaims;
  uint32_t max_page_reclaims;

  MDContent()
      : begin_ts(0), end_ts(0), min_rss(UINT32_MAX), max_rss(0), min_page_reclaims(UINT32_MAX),
        max_page_reclaims(0)
  {
    // DO NOTHING
  }

  virtual ~MDContent() = default;

  void updateRss(uint32_t rss)
  {
    if (min_rss == UINT32_MAX)
      min_rss = rss;
    if (max_rss == 0)
      max_rss = rss;

    if (min_rss > rss)
      min_rss = rss;
    else if (max_rss < rss)
      max_rss = rss;
  }

  void updateMinflt(uint32_t minflt)
  {
    if (min_page_reclaims == UINT32_MAX)
      min_page_reclaims = minflt;
    if (max_page_reclaims == 0)
      max_page_reclaims = minflt;

    if (min_page_reclaims > minflt)
      min_page_reclaims = minflt;
    else if (max_page_reclaims < minflt)
      max_page_reclaims = minflt;
  }

  virtual void write(std::ostream &os) const = 0;
};

struct OpSeq : public MDContent
{
  std::string backend;
  uint64_t graph_latency;

  struct OpSeqCmp
  {
    bool operator()(const OpSeq &lhs, const OpSeq &rhs) const
    {
      return lhs.begin_ts < rhs.begin_ts;
    }
    bool operator()(const OpSeq &lhs, const OpSeq &rhs) { return lhs.begin_ts < rhs.begin_ts; }
    bool operator()(OpSeq &lhs, OpSeq &rhs) { return lhs.begin_ts < rhs.begin_ts; }
  };

  void write(std::ostream &os) const override
  {
    uint64_t opseq_latency = end_ts - begin_ts;
    double opseq_per = static_cast<double>(opseq_latency) / graph_latency * 100.0;
    writeMDTableRow(os, {name, backend, std::to_string(opseq_latency), std::to_string(opseq_per),
                         std::to_string(min_rss), std::to_string(max_rss),
                         std::to_string(min_page_reclaims), std::to_string(max_page_reclaims)});
  }
};

struct Graph : public MDContent
{
  std::set<OpSeq, OpSeq::OpSeqCmp> opseqs;

  void setOpSeqs(const std::map<std::string, OpSeq> &name_to_opseq)
  {
    uint64_t graph_latency = end_ts - begin_ts;
    for (auto it : name_to_opseq)
    {
      auto opseq = it.second;
      opseq.graph_latency = graph_latency;

      opseqs.insert(opseq);

      updateRss(opseq.min_rss);
      updateRss(opseq.max_rss);
      updateMinflt(opseq.min_page_reclaims);
      updateMinflt(opseq.max_page_reclaims);
    }
  }

  void write(std::ostream &os) const override
  {
    static std::vector<std::string> graph_headers{"latency(us)", "rss_min(kb)", "rss_max(kb)",
                                                  "page_reclaims_min", "page_reclaims_max"};

    static std::vector<std::string> graph_headers_line{"-----------", "-------", "-------",
                                                       "-----------------", "-----------------"};

    // Graph's Header
    writeMDTableRow(os, graph_headers);
    writeMDTableRow(os, graph_headers_line);

    // Graph's contents
    writeMDTableRow(os, {std::to_string(end_ts - begin_ts), std::to_string(min_rss),
                         std::to_string(max_rss), std::to_string(min_page_reclaims),
                         std::to_string(max_page_reclaims)});

    os << "\n";

    static std::vector<std::string> opseq_headers{
        "OpSeq name",  "backend",     "latency(us)",       "latency(%)",
        "rss_min(kb)", "rss_max(kb)", "page_reclaims_min", "page_reclaims_max"};

    static std::vector<std::string> opseq_headers_line{
        "----------", "-------", "-----------",       "-----------",
        "-------",    "-------", "-----------------", "-----------------"};

    os << "## OpSequences \n";

    // OpSeq's Header
    writeMDTableRow(os, opseq_headers);
    writeMDTableRow(os, opseq_headers_line);

    // OpSeq's contents
    for (auto opseq : opseqs)
    {
      opseq.write(os);
    }

    os << "\n";
  }
};

struct MDTableBuilder
{
  MDTableBuilder(const std::vector<DurationEvent> &duration_events,
                 const std::vector<CounterEvent> &counter_events)
      : _duration_events(duration_events), _counter_events(counter_events)
  {
// when ready with low overhead in release build
#ifdef DEBUG
    for (const auto &evt : _counter_events)
    {
      uint64_t ts = std::stoull(evt.ts);
      auto &name = evt.name;
      assert(name.compare("maxrss") == 0 || name.compare("minflt") == 0);
      assert(evt.values.size() == 1);
      auto &val = evt.values.begin()->second;
      if (_ts_to_values.find(ts) == _ts_to_values.end())
      {
        std::pair<uint32_t, uint32_t> values;
        if (name.compare("maxrss") == 0)
          values.first = std::stoul(val);
        else
          values.second = std::stoul(val);
        _ts_to_values.insert({ts, values});
      }
      else
      {
        auto &values = _ts_to_values.at(ts);
        if (name.compare("maxrss") == 0)
          values.first = std::stoul(val);
        else
          values.second = std::stoul(val);
      }
    }
#endif
  }

  MDTableBuilder &build()
  {
    for (auto &it : divideGraph())
    {
      size_t begin_idx = it.first;
      size_t end_idx = it.second;
      std::map<std::string, OpSeq> name_to_opseq;
      for (size_t i = begin_idx + 1; i < end_idx; ++i)
      {
        const auto &evt = _duration_events[i];
        assert(evt.name.compare("Subgraph") != 0);
        assert(evt.ph.compare("B") == 0 || evt.ph.compare("E") == 0);
        if (evt.ph.compare("B") == 0)
        {
          assert(name_to_opseq.find(evt.name) == name_to_opseq.end());
          name_to_opseq.insert({evt.name, makeOpSeq(evt)});
        }
        else
        {
          assert(name_to_opseq.find(evt.name) != name_to_opseq.end());
          auto &opseq = name_to_opseq.at(evt.name);
          updateOpSeq(opseq, evt);
        }
      }

      _graphs.emplace_back(makeGraph(begin_idx, end_idx, name_to_opseq));
    }

    return *this;
  }

  std::vector<std::pair<size_t, size_t>> divideGraph()
  {
    std::vector<std::pair<size_t, size_t>> graph_idx_list; // pair<begin_idx, end_idx>
    for (size_t i = 0, begin_idx = 0; i < _duration_events.size(); ++i)
    {
      const auto &evt = _duration_events.at(i);
      if (evt.name.compare("Subgraph") == 0)
      {
        if (evt.ph.compare("B") == 0)
          begin_idx = i;
        else
          graph_idx_list.emplace_back(begin_idx, i);
      }
    }
    return graph_idx_list;
  }

  OpSeq makeOpSeq(const DurationEvent &evt)
  {
    OpSeq opseq;
    opseq.name = evt.name;
    opseq.begin_ts = std::stoull(evt.ts);
    opseq.backend = evt.tid;
#ifdef DEBUG
    opseq.updateRss(_ts_to_values.at(opseq.begin_ts).first);
    opseq.updateMinflt(_ts_to_values.at(opseq.begin_ts).second);
#else
    opseq.updateRss(0);
    opseq.updateMinflt(0);
#endif
    return opseq;
  }

  void updateOpSeq(OpSeq &opseq, const DurationEvent &evt)
  {
    opseq.end_ts = std::stoull(evt.ts);
#ifdef DEBUG
    opseq.updateRss(_ts_to_values.at(opseq.end_ts).first);
    opseq.updateMinflt(_ts_to_values.at(opseq.end_ts).second);
#else
    opseq.updateRss(0);
    opseq.updateMinflt(0);
#endif
  }

  Graph makeGraph(size_t begin_idx, size_t end_idx,
                  const std::map<std::string, OpSeq> &name_to_opseq)
  {
    Graph graph;
    graph.name = "Subgraph";
    graph.begin_ts = std::stoull(_duration_events[begin_idx].ts);
    graph.end_ts = std::stoull(_duration_events[end_idx].ts);
    graph.setOpSeqs(name_to_opseq);
#ifdef DEBUG
    graph.updateRss(_ts_to_values.at(graph.begin_ts).first);
    graph.updateMinflt(_ts_to_values.at(graph.begin_ts).second);
    graph.updateRss(_ts_to_values.at(graph.end_ts).first);
    graph.updateMinflt(_ts_to_values.at(graph.end_ts).second);
#else
    graph.updateRss(0);
    graph.updateMinflt(0);
#endif
    return graph;
  }

  void write(std::ostream &os)
  {
    // Write contents
    for (size_t i = 0; i < _graphs.size(); ++i)
    {
      os << "# Graph " << i << "\n";
      _graphs.at(i).write(os);
    }
  }

  const std::vector<DurationEvent> &_duration_events;
  const std::vector<CounterEvent> &_counter_events;
  // timestamp to std::pair<maxrss, minflt>
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> _ts_to_values;
  std::vector<Graph> _graphs;
};

} // namespace

EventWriter::EventWriter(const EventRecorder &recorder) : _recorder(recorder)
{
  // DO NOTHING
}

void EventWriter::writeToFiles(const std::string &base_filepath)
{
  // Note. According to an internal issue, let snpe json as just file name not '.snpe.json'
  writeToFile(base_filepath, WriteFormat::SNPE_BENCHMARK);
  writeToFile(base_filepath + ".chrome.json", WriteFormat::CHROME_TRACING);
  writeToFile(base_filepath + ".table.md", WriteFormat::MD_TABLE);
}

void EventWriter::writeToFile(const std::string &filepath, WriteFormat write_format)
{
  std::ofstream os{filepath, std::ofstream::out};
  switch (write_format)
  {
    case WriteFormat::CHROME_TRACING:
      writeChromeTrace(os);
      break;
    case WriteFormat::SNPE_BENCHMARK:
      writeSNPEBenchmark(os);
      break;
    case WriteFormat::MD_TABLE:
      writeMDTable(os);
      break;
    default:
      assert(!"Invalid value");
      break;
  }
}

void EventWriter::writeSNPEBenchmark(std::ostream &os)
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
    for (auto &evt : _recorder.counter_events())
    {
      auto &mem_stat = mem_stats[evt.name];
      uint64_t val = std::stoull(evt.values.at("value"));
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
    for (auto &evt : _recorder.duration_events())
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

void EventWriter::writeChromeTrace(std::ostream &os)
{
  os << "{\n";
  os << "  " << quote("traceEvents") << ": [\n";

  for (auto &evt : _recorder.duration_events())
  {
    os << "    " << object(evt) << ",\n";
  }

  for (auto &evt : _recorder.counter_events())
  {
    os << "    " << object(evt) << ",\n";
  }

  os << "    { }\n";
  os << "  ]\n";
  os << "}\n";
}

void EventWriter::writeMDTable(std::ostream &os)
{
  MDTableBuilder(_recorder.duration_events(), _recorder.counter_events()).build().write(os);
}
