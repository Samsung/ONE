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

#ifndef __STYLE_TRANSFER_APP_ARGS_H__
#define __STYLE_TRANSFER_APP_ARGS_H__

#include <arser/arser.h>

#include <string>

namespace StyleTransferApp
{

class Args
{
public:
  Args(const int argc, char **argv) noexcept;
  void print(void);

  const std::string &getPackageFilename(void) const { return _package_filename; }
  const std::string &getInputFilename(void) const { return _input_filename; }
  const std::string &getOutputFilename(void) const { return _output_filename; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  arser::Arser _arser{"style_transfer_app"};

  std::string _package_filename;
  std::string _input_filename;
  std::string _output_filename;
};

} // end of namespace StyleTransferApp

#endif // __STYLE_TRANSFER_APP_ARGS_H__
