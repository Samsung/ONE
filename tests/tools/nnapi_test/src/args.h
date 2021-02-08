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

#ifndef __NNAPI_TEST_ARGS_H__
#define __NNAPI_TEST_ARGS_H__

#include <boost/program_options.hpp>
#include <string>

namespace po = boost::program_options;

namespace nnapi_test
{

class Args
{
public:
  Args(const int argc, char **argv);
  void print(char **argv);

  const std::string &getTfliteFilename(void) const { return _tflite_filename; }
  const int getSeed(void) const { return _seed; }
  const int getNumRuns(void) const { return _num_runs; }
  const int getInputSet(void) const { return _input_set; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  po::positional_options_description _positional;
  po::options_description _options;

  std::string _tflite_filename;
  int _seed;
  int _num_runs;
  int _input_set;
};

} // end of namespace nnapi_test

#endif // __NNAPI_TEST_ARGS_H__
