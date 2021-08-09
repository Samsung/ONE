/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "pareto_lookup.h"
#include <iostream>
#include <fstream>
#include <json.h>

ParetoOptimizer::ParetoOptimizer(std::string json_file)
  : _json_file(json_file), _solution_map(), _backend_allocations()
{
  _threshold_margin_latency = 80;
  _threshold_margin_memory = 100;
}

void ParetoOptimizer::initialize_maps(void)
{
  std::ifstream cfg_file(_json_file, std::ifstream::binary);
  Json::CharReaderBuilder cfg;
  Json::Value root;
  Json::Value configs;
  Json::Value solutions;
  JSONCPP_STRING errs;
  cfg["collectComments"] = true;

  if (!parseFromStream(cfg, cfg_file, &root, &errs))
  {
    std::cout << errs << std::endl;
    return;
  }
  configs = root["configs"];
  for (auto it = configs.begin(); it != configs.end(); ++it)
  {
    _backend_allocations[it.key().asString()] = (*it).asString();
  }
  solutions = root["solutions"];
  for (auto it = solutions.begin(); it != solutions.end(); ++it)
  {
    auto solution = *it;
    float exec_time;
    float max_rss;
    _largest_rss_value = 0;
    std::string id;
    for (auto itr = solution.begin(); itr != solution.end(); ++itr)
    {
      if (itr.key() == "id")
      {
        id = (*itr).asString();
      }
      else if (itr.key() == "exec_time")
      {
        exec_time = (*itr).asFloat();
      }
      else if (itr.key() == "max_rss")
      {
        max_rss = (*itr).asFloat();
        if (_largest_rss_value < max_rss)
        {
          _largest_rss_value = max_rss;
        }
      }
    }
    _solution_map[id] = std::make_tuple(exec_time, max_rss);
  }
  cfg_file.close();
}

void ParetoOptimizer::print_maps(void)
{
  for (auto x : _solution_map)
  {
    std::cout << x.first << " : (" << std::get<0>(x.second) << "," << std::get<1>(x.second) << ")"
              << std::endl;
  }
  for (auto x : _backend_allocations)
  {
    std::cout << x.first << " : " << x.second << std::endl;
  }
}
float ParetoOptimizer::fetch_smallest_exec(void)
{
  float minval = 999999;
  std::string min_id;
  std::string config;
  for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
  {
    auto exec_time = std::get<0>(it->second);
    if (exec_time < minval)
    {
      minval = exec_time;
    }
  }
  return minval;
}

std::string ParetoOptimizer::fetch_config_with_smallest_exec(void)
{
  float minval = 999999;
  std::string min_id;
  std::string config;
  for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
  {
    auto exec_time = std::get<0>(it->second);
    if (exec_time < minval)
    {
      minval = exec_time;
      min_id = it->first;
    }
  }
  _current_reference_time = minval;
  _threshold_time_check = _threshold_margin_latency * _current_reference_time / 100;
  _current_reference_memory = std::get<1>(_solution_map[min_id]);
  _threshold_memory_check = _threshold_margin_memory * _current_reference_memory / 100;
  config = _backend_allocations[min_id];
  return config;
}
std::string ParetoOptimizer::fetch_config_with_smallest_memory(void)
{
  float minval = 9999999;
  std::string min_id;
  std::string config;
  for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
  {
    auto max_rss = std::get<1>(it->second);
    if (max_rss < minval)
    {
      minval = max_rss;
      min_id = it->first;
    }
  }
  _current_reference_time = std::get<0>(_solution_map[min_id]);
  _threshold_time_check = _threshold_margin_latency * _current_reference_time / 100;
  _current_reference_memory = minval;
  _threshold_memory_check = _threshold_margin_memory * _current_reference_memory / 100;
  config = _backend_allocations[min_id];
  return config;
}

std::string ParetoOptimizer::fetch_config_within_exectime(float exec_time_limit)
{
  float min_difference = 999999;
  float difference;
  std::string min_id;
  std::string config;
  for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
  {
    auto exec_time = std::get<0>(it->second);
    if (exec_time < exec_time_limit)
    {
      difference = exec_time_limit - exec_time;
      if (difference < min_difference)
      {
        min_difference = difference;
        min_id = it->first;
      }
    }
  }
  if (_current_reference_time == std::get<0>(_solution_map[min_id]))
  {
    return std::string();
  }
  _current_reference_time = std::get<0>(_solution_map[min_id]);
  _threshold_time_check = _threshold_margin_latency * _current_reference_time / 100;
  _current_reference_memory = std::get<1>(_solution_map[min_id]);
  _threshold_memory_check = _threshold_margin_memory * _current_reference_memory / 100;
  config = _backend_allocations[min_id];
  return config;
}

std::string ParetoOptimizer::fetch_config_within_memory(int32_t memory_diff)
{
  float max_val = 0;
  std::string min_id;
  std::string config;
  for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
  {
    auto max_rss = std::get<1>(it->second);
    if ((max_rss <= (_current_reference_memory + memory_diff)) && (max_rss > max_val))
    {
      max_val = max_rss;
      min_id = it->first;
    }
  }
  _current_reference_time = std::get<0>(_solution_map[min_id]);
  _threshold_time_check = _threshold_margin_latency * _current_reference_time / 100;
  _current_reference_memory = std::get<1>(_solution_map[min_id]);
  _threshold_memory_check = _threshold_margin_memory * _current_reference_memory / 100;
  config = _backend_allocations[min_id];
  return config;
}

std::string ParetoOptimizer::fetch_config_within_memory_bound(uint32_t currently_available_memory)
{
  int32_t max_val = 0;
  std::string min_id;
  std::string config;
  std::cout << "arg: " << std::to_string(currently_available_memory) << std::endl;
  for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
  {
    auto max_rss = std::get<1>(it->second);
    std::cout << "max_rss = " << std::to_string(max_rss) << std::endl;
    if ((max_rss <= currently_available_memory) && (max_rss > max_val))
    {
      max_val = max_rss;
      min_id = it->first;
    }
  }
  std::cout << "min_id = " << min_id << ", max_val = " << std::to_string(max_val) << std::endl;
  _current_reference_time = std::get<0>(_solution_map[min_id]);
  _threshold_time_check = _threshold_margin_latency * _current_reference_time / 100;
  _current_reference_memory = std::get<1>(_solution_map[min_id]);
  _threshold_memory_check = _threshold_margin_memory * _current_reference_memory / 100;
  std::cout << std::to_string(_current_reference_time) << ", "
            << std::to_string(_current_reference_memory) << std::endl;
  config = _backend_allocations[min_id];
  return config;
}

bool ParetoOptimizer::exec_time_increased(float exec_time)
{
  return ((exec_time > _current_reference_time) &&
          (exec_time - _current_reference_time) > _threshold_time_check);
}

bool ParetoOptimizer::feasible_memory_increase(int32_t memory_diff)
{
  if ((memory_diff > _threshold_memory_check) && (_current_reference_memory != _largest_rss_value))
    return true;
  return false;
}

std::string ParetoOptimizer::get_current_setting(void)
{
  return "(" + std::to_string(_current_reference_time) + ", " +
         std::to_string(_current_reference_memory) + ") ";
}

int32_t ParetoOptimizer::get_pareto_rss(void) { return _current_reference_memory; }

void ParetoOptimizer::set_thresholds(int margin_latency, int margin_memory)
{
  _threshold_margin_latency = margin_latency;
  _threshold_margin_memory = margin_memory;
}
