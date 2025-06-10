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

#ifndef __TFLITE_LOADER_TOOLS_SRC_ARGS_H__
#define __TFLITE_LOADER_TOOLS_SRC_ARGS_H__

#include <string>
#include <arser/arser.h>

namespace TFLiteRun
{

class Args
{
public:
  Args(const int argc, char **argv) noexcept;
  void print(char **argv);

  const std::string &getTFLiteFilename(void) const { return _tflite_filename; }
  const std::vector<std::string> &getDataFilenames(void) const { return _data_filenames; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  arser::Arser _arser;

  std::string _tflite_filename;
  std::vector<std::string> _data_filenames;
};

} // namespace TFLiteRun

#endif // __TFLITE_LOADER_TOOLS_SRC_ARGS_H__
