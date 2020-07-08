/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFLITE_RUN_ARGS_H__
#define __TFLITE_RUN_ARGS_H__

#include <string>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace TFLiteRun
{

class Args
{
public:
  Args(const int argc, char **argv) noexcept;
  void print(void);

  const std::string &getTFLiteFilename(void) const { return _tflite_filename; }
  const std::string &getDumpFilename(void) const { return _dump_filename; }
  const std::string &getCompareFilename(void) const { return _compare_filename; }
  const std::string &getInputFilename(void) const { return _input_filename; }
  const std::vector<int> &getInputShapes(void) const { return _input_shapes; }
  const int getNumRuns(void) const { return _num_runs; }
  const int getWarmupRuns(void) const { return _warmup_runs; }
  const int getRunDelay(void) const { return _run_delay; }
  const bool getGpuMemoryPoll(void) const { return _gpumem_poll; }
  const bool getMemoryPoll(void) const { return _mem_poll; }
  const bool getWriteReport(void) const { return _write_report; }
  const bool getModelValidate(void) const { return _tflite_validate; }
  const int getVerboseLevel(void) const { return _verbose_level; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  po::positional_options_description _positional;
  po::options_description _options;

  std::string _tflite_filename;
  std::string _dump_filename;
  std::string _compare_filename;
  std::string _input_filename;
  std::vector<int> _input_shapes;
  int _num_runs;
  int _warmup_runs;
  int _run_delay;
  bool _gpumem_poll;
  bool _mem_poll;
  bool _write_report;
  bool _tflite_validate;
  int _verbose_level;
};

} // end of namespace TFLiteRun

#endif // __TFLITE_RUN_ARGS_H__
