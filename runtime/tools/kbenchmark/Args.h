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

#ifndef __KBENCHMARK_ARGS_H__
#define __KBENCHMARK_ARGS_H__

#include <string>
#include <vector>
namespace kbenchmark
{

class Args
{
public:
  Args(const int argc, char **argv) noexcept;

  const std::string &config(void) { return _config; }
  const std::vector<std::string> &kernel(void) { return _kernel; }
  const std::string &reporter(void) { return _reporter; }
  const std::string &filter(void) { return _filter; }
  const std::string &output(void) { return _output; }
  int verbose(void) { return _verbose; }

private:
  void Initialize(const int argc, char **argv);

private:
  std::string _config;
  std::vector<std::string> _kernel;
  std::string _reporter;
  std::string _filter;
  std::string _output;
  int _verbose;
};

} // namespace kbenchmark

#endif // __KBENCHMARK_ARGS_H__
