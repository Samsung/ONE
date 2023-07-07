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
#include <map>
#include <set>
#include <sstream>
#include <stdint.h>
#include <unordered_map>
#include <utility>
#include <vector>

// md table type
namespace
{

void writeMDTableRow(std::ostream &os, const std::vector<std::string> &list)
{
  os << "| ";
  for (const auto &key : list)
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

struct Operation : public MDContent
{
  std::string backend;
  uint64_t graph_latency;

  struct OperationCmp
  {
    bool operator()(const Operation &lhs, const Operation &rhs) const
    {
      return lhs.begin_ts < rhs.begin_ts;
    }
    bool operator()(const Operation &lhs, const Operation &rhs)
    {
      return lhs.begin_ts < rhs.begin_ts;
    }
    bool operator()(Operation &lhs, Operation &rhs) { return lhs.begin_ts < rhs.begin_ts; }
  };

  void write(std::ostream &os) const override
  {
    uint64_t op_latency = end_ts - begin_ts;
    double op_per = static_cast<double>(op_latency) / graph_latency * 100.0;
    writeMDTableRow(os, {name, backend, std::to_string(op_latency), std::to_string(op_per),
                         std::to_string(min_rss), std::to_string(max_rss),
                         std::to_string(min_page_reclaims), std::to_string(max_page_reclaims)});
  }
};

struct Graph : public MDContent
{
  std::set<Operation, Operation::OperationCmp> ops;
  std::string session_index;
  std::string subgraph_index;

  void setOperations(const std::map<std::string, Operation> &name_to_op)
  {
    uint64_t graph_latency = end_ts - begin_ts;
    for (auto &&it : name_to_op)
    {
      auto op = it.second;
      op.graph_latency = graph_latency;

      ops.insert(op);

      updateRss(op.min_rss);
      updateRss(op.max_rss);
      updateMinflt(op.min_page_reclaims);
      updateMinflt(op.max_page_reclaims);
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

    static std::vector<std::string> op_headers{
      "Op name",     "backend",     "latency(us)",       "latency(%)",
      "rss_min(kb)", "rss_max(kb)", "page_reclaims_min", "page_reclaims_max"};

    static std::vector<std::string> op_headers_line{
      "-------", "-------", "-----------",       "-----------",
      "-------", "-------", "-----------------", "-----------------"};

    os << "## Op \n";

    // Operation's Header
    writeMDTableRow(os, op_headers);
    writeMDTableRow(os, op_headers_line);

    // Operation's contents
    for (auto &&op : ops)
    {
      op.write(os);
    }

    os << "\n";
  }
};

std::string getLabel(const OpSeqDurationEvent &evt)
{
  std::string subg_label("$" + std::to_string(evt.subg_index) + " subgraph");
  std::string op_label("@" + std::to_string(evt.op_index) + " " + evt.op_name);

  return subg_label + " " + op_label;
}

struct MDTableBuilder
{
  MDTableBuilder(const std::vector<std::unique_ptr<DurationEvent>> &duration_events,
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
    for (const auto &it : divideGraph())
    {
      size_t begin_idx = it.first;
      size_t end_idx = it.second;
      std::map<std::string, Operation> name_to_op;
      for (size_t i = begin_idx + 1; i < end_idx; ++i)
      {
        const auto *evt = dynamic_cast<const OpSeqDurationEvent *>(_duration_events[i].get());
        if (evt == nullptr)
          continue;

        const std::string evt_name = getLabel(*evt);
        assert(evt->ph.compare("B") == 0 || evt->ph.compare("E") == 0);
        if (evt->ph.compare("B") == 0)
        {
          assert(name_to_op.find(evt_name) == name_to_op.end());
          name_to_op.insert({evt_name, makeOperation(*evt)});
        }
        else
        {
          assert(name_to_op.find(evt_name) != name_to_op.end());
          auto &op = name_to_op.at(evt_name);
          updateOperation(op, *evt);
        }
      }

      _graphs.emplace_back(makeGraph(begin_idx, end_idx, name_to_op));
    }

    return *this;
  }

  std::vector<std::pair<size_t, size_t>> divideGraph()
  {
    std::vector<std::pair<size_t, size_t>> graph_idx_list; // pair<begin_idx, end_idx>
    for (size_t i = 0, begin_idx = 0; i < _duration_events.size(); ++i)
    {
      const auto subg_evt = dynamic_cast<const SubgDurationEvent *>(_duration_events.at(i).get());
      if (subg_evt == nullptr)
        continue;

      if (subg_evt->ph.compare("B") == 0)
        begin_idx = i;
      else
        graph_idx_list.emplace_back(begin_idx, i);
    }
    return graph_idx_list;
  }

  Operation makeOperation(const OpSeqDurationEvent &evt)
  {
    Operation op;
    const std::string &evt_name = getLabel(evt);
    op.name = evt_name;
    op.begin_ts = std::stoull(evt.ts);
    op.backend = evt.backend;
#ifdef DEBUG
    op.updateRss(_ts_to_values.at(op.begin_ts).first);
    op.updateMinflt(_ts_to_values.at(op.begin_ts).second);
#else
    op.updateRss(0);
    op.updateMinflt(0);
#endif
    return op;
  }

  void updateOperation(Operation &op, const DurationEvent &evt)
  {
    op.end_ts = std::stoull(evt.ts);
#ifdef DEBUG
    op.updateRss(_ts_to_values.at(op.end_ts).first);
    op.updateMinflt(_ts_to_values.at(op.end_ts).second);
#else
    op.updateRss(0);
    op.updateMinflt(0);
#endif
  }

  Graph makeGraph(size_t begin_idx, size_t end_idx,
                  const std::map<std::string, Operation> &name_to_op)
  {
    Graph graph;
    graph.name = "Subgraph";
    graph.begin_ts = std::stoull(_duration_events[begin_idx]->ts);
    graph.end_ts = std::stoull(_duration_events[end_idx]->ts);
    graph.setOperations(name_to_op);

    for (const auto &arg : _duration_events[end_idx]->args)
    {
      if (arg.first == "session")
        graph.session_index = arg.second;
      if (arg.first == "subgraph")
        graph.subgraph_index = arg.second;
    }

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
      auto &graph = _graphs.at(i);
      os << "# Session: " << graph.session_index << ", Subgraph: " << graph.subgraph_index
         << ", Running count: " << i << "\n";
      _graphs.at(i).write(os);
    }
  }

  const std::vector<std::unique_ptr<DurationEvent>> &_duration_events;
  const std::vector<CounterEvent> &_counter_events;

  // timestamp to std::pair<maxrss, minflt>
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> _ts_to_values;
  std::vector<Graph> _graphs;
};

} // namespace

void MDTableWriter::flush(const std::vector<std::unique_ptr<EventRecorder>> &records)
{
  for (const auto &recorder : records)
  {
    MDTableBuilder(recorder->duration_events(), recorder->counter_events()).build().write(_os);
  }
}
