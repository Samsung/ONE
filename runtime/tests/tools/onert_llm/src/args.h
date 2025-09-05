/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_LLM_ARGS_H__
#define __ONERT_LLM_ARGS_H__

#include <string>
#include <unordered_map>
#include <vector>

#include <arser/arser.h>

namespace onert_llm
{

class Args
{
public:
  Args(const int argc, char **argv);
  void print(void);

  const std::string &getPackageFilename(void) const { return _package_filename; }
  const std::string &getDumpRawFilename(void) const { return _dump_raw_filename; }
  const std::string &getLoadRawFilename(void) const { return _load_raw_filename; }
  bool printVersion(void) const { return _print_version; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  arser::Arser _arser;

  std::string _package_filename;
  std::string _dump_raw_filename;
  std::string _load_raw_filename;
  bool _print_version = false;
};

} // end of namespace onert_llm

#endif // __ONERT_LLM_ARGS_H__
