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

#ifndef __NNPACKAGE_RUN_ARGS_H__
#define __NNPACKAGE_RUN_ARGS_H__

#include <string>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace nnpkg_run
{

class Args
{
public:
  Args(const int argc, char **argv) noexcept;
  void print(void);

  const std::string &getPackageFilename(void) const { return _package_filename; }
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
  const std::string &getDumpFilename(void) const { return _dump_filename; }
  const std::string &getLoadFilename(void) const { return _load_filename; }
#endif
  const int getNumRuns(void) const { return _num_runs; }
  const int getWarmupRuns(void) const { return _warmup_runs; }
  std::unordered_map<uint32_t, uint32_t> getOutputSizes(void) const { return _output_sizes; }
  const bool getGpuMemoryPoll(void) const { return _gpumem_poll; }
  const bool getMemoryPoll(void) const { return _mem_poll; }
  const bool getWriteReport(void) const { return _write_report; }
  const bool printVersion(void) const { return _print_version; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  po::positional_options_description _positional;
  po::options_description _options;

  std::string _package_filename;
#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
  std::string _dump_filename;
  std::string _load_filename;
#endif
  int _num_runs;
  int _warmup_runs;
  std::unordered_map<uint32_t, uint32_t> _output_sizes;
  bool _gpumem_poll;
  bool _mem_poll;
  bool _write_report;
  bool _print_version = false;
};

} // end of namespace nnpkg_run

#endif // __NNPACKAGE_RUN_ARGS_H__
