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

#include <iostream>
#include <fstream>
#include "common.h"
#include "session.h"
#include "memory_stats.h"
#include "nnfw_experimental.h"

uint64_t RunSession::num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    n *= ti->dims[i];
  }
  return n;
}
template <typename T> void RunSession::random_input_float(float *vec, T nelements)
{
  T i;
  for (i = 0; i < nelements; i++)
  {
    vec[i] = (((rand() % 100) + 1) / 1000.f);
  }
}

template <typename T1, typename T2> void RunSession::random_input_int(T1 *vec, T2 nelements)
{
  T2 i;
  for (i = 0; i < nelements; i++)
  {
    vec[i] = rand() % 100 + 1;
  }
}

RunSession::RunSession(std::string model)
  : _model(model), _session(nullptr), _inputs(), _inputs_initialized(false)
{
}

void RunSession::load_session(void)
{
  struct rusage res_usage_begin;
  struct rusage res_usage_end;
  getrusage(RUSAGE_SELF, &res_usage_begin);
  nnfw_create_session(&_session);

  std::string best_config;
  best_config = opt->fetch_config_with_smallest_exec();
  auto currently_available_memory = get_meminfo(MEM_AVAILABLE);
  if (opt->get_pareto_rss() > currently_available_memory)
  {
    std::cout << "Available memory = " << std::to_string(currently_available_memory) << std::endl;
    best_config = opt->fetch_config_within_memory_bound(currently_available_memory * 2 / 3);
    std::cout << "after initial reconfig: " << opt->get_current_setting() << std::endl;
  }
  std::string pattern = "OP_BACKEND_MAP=\"";
  std::string backend_setting =
    best_config.substr(best_config.find(pattern) + pattern.size(),
                       best_config.size() - 1 - (best_config.find(pattern) + pattern.size()));
  std::cout << "Best Configuration: " << backend_setting << std::endl;

  // Loading nnpackage
  nnfw_load_model_from_file(_session, _model.c_str());
  nnfw_set_backends_per_operation(_session, backend_setting.c_str());

  // Compile model
  nnfw_prepare(_session);
  prepare_output();
  getrusage(RUSAGE_SELF, &res_usage_end);
  json->add_instance_record("Initial setting: (" + get_pareto_setting() + "), RSS increase: (" +
                            std::to_string(res_usage_end.ru_maxrss) + ", " +
                            std::to_string(res_usage_begin.ru_maxrss) + ")");
}

void RunSession::reconfigure_for_smallest_exec(void)
{
  std::string setting_old = get_pareto_setting();
  std::string best_config;
  best_config = opt->fetch_config_with_smallest_exec();
  auto currently_available_memory = get_meminfo(MEM_AVAILABLE);
  if (opt->get_pareto_rss() > currently_available_memory)
  {
    std::cout << "Available memory = " << std::to_string(currently_available_memory) << std::endl;
    best_config = opt->fetch_config_within_memory_bound(currently_available_memory * 2 / 3);
    std::cout << "after initial reconfig: " << opt->get_current_setting() << std::endl;
  }
  std::string pattern = "OP_BACKEND_MAP=\"";
  std::string backend_setting =
    best_config.substr(best_config.find(pattern) + pattern.size(),
                       best_config.size() - 1 - (best_config.find(pattern) + pattern.size()));
  json->add_timed_record("session reconfig", "B");
  auto mem1 = get_meminfo(MEM_AVAILABLE);
  auto free1 = get_meminfo(MEM_FREE);
  double rss1, vm1;
  double rss2, vm2;

  process_mem_usage(vm1, rss1);
  nnfw_close_session(_session);
  nnfw_create_session(&_session);
  nnfw_load_model_from_file(_session, _model.c_str());
  nnfw_set_backends_per_operation(_session, backend_setting.c_str());
  nnfw_prepare(_session);
  prepare_output();
  process_mem_usage(vm2, rss2);
  auto mem2 = get_meminfo(MEM_AVAILABLE);
  auto free2 = get_meminfo(MEM_FREE);
  json->add_instance_record(setting_old + " --> " + get_pareto_setting() + "meminfo increase: (" +
                            std::to_string(mem1) + ":" + std::to_string(free1) + ", " +
                            std::to_string(mem2) + ":" + std::to_string(free2) + ")");
  json->add_instance_record("controller RSS (b4, after) : " + std::to_string(rss1) + ":" +
                            std::to_string(rss2));
  json->add_timed_record("session reconfig", "E");
}

bool RunSession::latency_increased(float exec_time) { return opt->exec_time_increased(exec_time); }

bool RunSession::memory_improved(int32_t memory_diff)
{
  return opt->feasible_memory_increase(memory_diff);
}

bool RunSession::reconfigure_within_exec_time(float exec_time)
{
  double vm1, rss1;
  double vm2, rss2;
  std::string setting_old = get_pareto_setting();
  std::string best_config = opt->fetch_config_within_exectime(exec_time);
  if (best_config.empty())
  {
    return false;
  }
  std::string pattern = "OP_BACKEND_MAP=\"";
  std::string backend_setting =
    best_config.substr(best_config.find(pattern) + pattern.size(),
                       best_config.size() - 1 - (best_config.find(pattern) + pattern.size()));
  json->add_timed_record("session reconfig", "B");
  auto mem1 = get_meminfo(MEM_AVAILABLE);
  auto free1 = get_meminfo(MEM_FREE);
  process_mem_usage(vm1, rss1);
  nnfw_close_session(_session);

  // Re-open session
  nnfw_create_session(&_session);
  nnfw_load_model_from_file(_session, _model.c_str());
  nnfw_set_backends_per_operation(_session, backend_setting.c_str());
  nnfw_prepare(_session);
  prepare_output();

  auto mem2 = get_meminfo(MEM_AVAILABLE);
  auto free2 = get_meminfo(MEM_FREE);
  process_mem_usage(vm1, rss2);
  json->add_instance_record(
    setting_old + " -> Alert increased time -> " + get_pareto_setting() + "meminfo increase: (" +
    std::to_string(mem1) + ", " + std::to_string(free1) + ":" + std::to_string(rss1) + ", " +
    std::to_string(mem2) + ", " + std::to_string(free2) + ":" + std::to_string(rss2) + ")");
  json->add_timed_record("session reconfig", "E");
  return true;
}

void RunSession::reconfigure_within_memory(int32_t memory_val)
{
  std::string setting_old = get_pareto_setting();
  std::string best_config = opt->fetch_config_within_memory(memory_val);
  std::string pattern = "OP_BACKEND_MAP=\"";
  std::string backend_setting =
    best_config.substr(best_config.find(pattern) + pattern.size(),
                       best_config.size() - 1 - (best_config.find(pattern) + pattern.size()));
  json->add_timed_record("session reconfig", "B");
  auto mem1 = get_meminfo(MEM_AVAILABLE);
  auto free1 = get_meminfo(MEM_FREE);
  double rss1, vm1;
  double rss2, vm2;

  process_mem_usage(vm1, rss1);
  nnfw_close_session(_session);
  nnfw_create_session(&_session);
  nnfw_load_model_from_file(_session, _model.c_str());
  nnfw_set_backends_per_operation(_session, backend_setting.c_str());
  nnfw_prepare(_session);
  prepare_output();
  process_mem_usage(vm2, rss2);
  auto mem2 = get_meminfo(MEM_AVAILABLE);
  auto free2 = get_meminfo(MEM_FREE);
  json->add_instance_record(setting_old + " --> " + get_pareto_setting() + "meminfo increase: (" +
                            std::to_string(mem1) + ":" + std::to_string(free1) + ", " +
                            std::to_string(mem2) + ":" + std::to_string(free2) + ")");
  json->add_instance_record("controller RSS (b4, after) : " + std::to_string(rss1) + ":" +
                            std::to_string(rss2));
  json->add_timed_record("session reconfig", "E");
}

int64_t RunSession::run_inference(void)
{
  int64_t st_time;
  int64_t end_time;
  st_time = json->add_timed_record("session run", "B");
  nnfw_run(_session);
  end_time = json->add_timed_record("session run", "E");
  return (end_time - st_time);
}

void RunSession::close(void) { nnfw_close_session(_session); }

void RunSession::initialize_inputs(std::ifstream &ifile)
{
  uint32_t n_inputs;
  nnfw_tensorinfo ti;
  nnfw_input_size(_session, &n_inputs);
  for (auto i = 0; i < n_inputs; i++)
  {
    nnfw_input_tensorinfo(_session, i, &ti);
    uint32_t input_elements = num_elems(&ti);
    switch (ti.dtype)
    {
      case NNFW_TYPE_TENSOR_FLOAT32:
      {
        float *input;
        if (_inputs_initialized == false)
        {
          input = new float[input_elements];
          _inputs.emplace_back(static_cast<void *>(input));
        }
        else
        {
          input = static_cast<float *>(_inputs[i]);
        }
        ifile.read(reinterpret_cast<char *>(input), input_elements * sizeof(float));
        nnfw_set_input(_session, i, ti.dtype, input, sizeof(float) * input_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_INT32:
      {
        int32_t *input;
        if (_inputs_initialized == false)
        {
          input = new int32_t[input_elements];
          _inputs.emplace_back(static_cast<void *>(input));
        }
        else
        {
          input = static_cast<int32_t *>(_inputs[i]);
        }
        ifile.read(reinterpret_cast<char *>(input), input_elements * sizeof(int32_t));
        nnfw_set_input(_session, i, ti.dtype, input, sizeof(int32_t) * input_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
      case NNFW_TYPE_TENSOR_BOOL:
      case NNFW_TYPE_TENSOR_UINT8:
      {
        uint8_t *input;
        if (_inputs_initialized == false)
        {
          input = new uint8_t[input_elements];
          _inputs.emplace_back(static_cast<void *>(input));
        }
        else
        {
          input = static_cast<uint8_t *>(_inputs[i]);
        }
        ifile.read(reinterpret_cast<char *>(input), input_elements * sizeof(uint8_t));
        nnfw_set_input(_session, i, ti.dtype, input, sizeof(uint8_t) * input_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
      {
        int8_t *input;
        if (_inputs_initialized == false)
        {
          input = new int8_t[input_elements];
          _inputs.emplace_back(static_cast<void *>(input));
        }
        else
        {
          input = static_cast<int8_t *>(_inputs[i]);
        }
        ifile.read(reinterpret_cast<char *>(input), input_elements * sizeof(int8_t));
        nnfw_set_input(_session, i, ti.dtype, input, sizeof(int8_t) * input_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_INT64:
      {
        int64_t *input;
        if (_inputs_initialized == false)
        {
          input = new int64_t[input_elements];
          _inputs.emplace_back(static_cast<void *>(input));
        }
        else
        {
          input = static_cast<int64_t *>(_inputs[i]);
        }
        ifile.read(reinterpret_cast<char *>(input), input_elements * sizeof(int64_t));
        nnfw_set_input(_session, i, ti.dtype, input, sizeof(int64_t) * input_elements);
        break;
      }
      default:
        std::cout << "Uknown input data type " << ti.dtype << std::endl;
        break;
    }
  }
  if (_inputs_initialized == false)
  {
    _inputs_initialized = true;
  }
}

void RunSession::prepare_output(void)
{
  uint32_t n_outputs;
  nnfw_tensorinfo ti;
  nnfw_output_size(_session, &n_outputs);
  for (auto i = 0; i < n_outputs; i++)
  {
    nnfw_output_tensorinfo(_session, i, &ti);
    uint32_t output_elements = num_elems(&ti);
    switch (ti.dtype)
    {
      case NNFW_TYPE_TENSOR_FLOAT32:
      {
        float *output = new float[output_elements];
        nnfw_set_output(_session, i, ti.dtype, output, sizeof(float) * output_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_INT32:
      {
        int32_t *output = new int32_t[output_elements];
        nnfw_set_output(_session, i, ti.dtype, output, sizeof(int32_t) * output_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
      case NNFW_TYPE_TENSOR_BOOL:
      case NNFW_TYPE_TENSOR_UINT8:
      {
        uint8_t *output = new uint8_t[output_elements];
        nnfw_set_output(_session, i, ti.dtype, output, sizeof(uint8_t) * output_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
      {
        int8_t *output = new int8_t[output_elements];
        nnfw_set_output(_session, i, ti.dtype, output, sizeof(int8_t) * output_elements);
        break;
      }
      case NNFW_TYPE_TENSOR_INT64:
      {
        int64_t *output = new int64_t[output_elements];
        nnfw_set_output(_session, i, ti.dtype, output, sizeof(int64_t) * output_elements);
        break;
      }
      default:
        std::cout << "Uknown output data type " << ti.dtype << std::endl;
        break;
    }
  }
}

std::string RunSession::get_pareto_setting(void) { return opt->get_current_setting(); }

std::string RunSession::get_model(void) { return _model; }

void RunSession::prepare_bulk_data(int n_iterations)
{
  uint32_t n_inputs;
  nnfw_tensorinfo ti;
  nnfw_input_size(_session, &n_inputs);

  for (auto i = 0; i < n_inputs; i++)
  {
    std::string basename = _model.substr(_model.find_last_of("/\\") + 1);
    std::string filename = "/tmp/bulkdata_" + basename + "_" + std::to_string(i) + ".dat";
    std::ofstream ofile(filename, std::ios::binary);
    nnfw_input_tensorinfo(_session, i, &ti);
    uint32_t input_elements = num_elems(&ti);
    for (auto n = 0; n < n_iterations; n++)
    {
      switch (ti.dtype)
      {
        case NNFW_TYPE_TENSOR_FLOAT32:
        {
          float *input;
          if (_inputs_initialized == false)
          {
            input = new float[input_elements];
            _inputs.emplace_back(static_cast<void *>(input));
          }
          else
          {
            input = static_cast<float *>(_inputs[i]);
          }

          random_input_float(input, input_elements);
          ofile.write(reinterpret_cast<char *>(input), input_elements * sizeof(float));
          break;
        }
        case NNFW_TYPE_TENSOR_INT32:
        {
          int32_t *input;
          random_input_int(input, input_elements);
          if (_inputs_initialized == false)
          {
            input = new int32_t[input_elements];
            _inputs.emplace_back(static_cast<void *>(input));
          }
          else
          {
            input = static_cast<int32_t *>(_inputs[i]);
          }
          random_input_int<int32_t, uint32_t>(input, input_elements);
          ofile.write(reinterpret_cast<char *>(input), input_elements * sizeof(int32_t));
          break;
        }
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
        case NNFW_TYPE_TENSOR_BOOL:
        case NNFW_TYPE_TENSOR_UINT8:
        {
          uint8_t *input;
          if (_inputs_initialized == false)
          {
            input = new uint8_t[input_elements];
            _inputs.emplace_back(static_cast<void *>(input));
          }
          else
          {
            input = static_cast<uint8_t *>(_inputs[i]);
          }
          random_input_int<uint8_t, uint32_t>(input, input_elements);
          ofile.write(reinterpret_cast<char *>(input), input_elements * sizeof(uint8_t));
          break;
        }
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
        {
          int8_t *input;
          if (_inputs_initialized == false)
          {
            int8_t *input = new int8_t[input_elements];
            _inputs.emplace_back(static_cast<void *>(input));
          }
          else
          {
            input = static_cast<int8_t *>(_inputs[i]);
          }
          random_input_int<int8_t, uint32_t>(input, input_elements);
          ofile.write(reinterpret_cast<char *>(input), input_elements * sizeof(int8_t));
          break;
        }
        case NNFW_TYPE_TENSOR_INT64:
        {
          int64_t *input;
          if (_inputs_initialized == false)
          {
            int64_t *input = new int64_t[input_elements];
            _inputs.emplace_back(static_cast<void *>(input));
          }
          else
          {
            input = static_cast<int64_t *>(_inputs[i]);
          }
          random_input_int<int64_t, uint32_t>(input, input_elements);
          ofile.write(reinterpret_cast<char *>(input), input_elements * sizeof(int64_t));
          break;
        }
        default:
          std::cout << "Uknown input data type " << ti.dtype << std::endl;
          break;
      }
      if (_inputs_initialized == false)
      {
        _inputs_initialized = true;
      }
    }
    ofile.close();
  }
}
