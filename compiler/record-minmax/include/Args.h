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

#ifndef __RECORD_MINMAX_ARGS_H__
#define __RECORD_MINMAX_ARGS_H__

#include <string>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace record_minmax
{

class Args
{
public:
  Args(const int argc, char **argv);
  void print(void);

  const std::string &getInputModelFilename(void) const { return _input_model_filename; }
  const std::string &getInputDataFilename(void) const { return _input_data_filename; }
  const std::string &getOutputModelFilename(void) const { return _output_model_filename; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  po::positional_options_description _positional;
  po::options_description _options;

  std::string _input_model_filename;
  std::string _input_data_filename;
  std::string _output_model_filename;
};

} // end of namespace record_minmax

#endif // __RECORD_MINMAX_ARGS_H__
