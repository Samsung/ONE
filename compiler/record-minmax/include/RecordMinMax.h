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

#ifndef __RECORD_MINMAX_H__
#define __RECORD_MINMAX_H__

#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>

#include "DataSetIterator.h"
#include "MinMaxObserver.h"
#include "MinMaxComputer.h"

#include <memory>
#include <thread>

namespace record_minmax
{

using Buffer = std::vector<char>;
using Output = std::vector<Buffer>;
using WholeOutput = std::vector<Output>;

class RecordMinMax
{
public:
  enum DataSetFormat
  {
    UNKNOWN, // To check if format is set properly
    RANDOM,
    H5,
    DIRECTORY,
    LIST_FILE,
  };

  explicit RecordMinMax(uint32_t num_threads, std::unique_ptr<MinMaxComputer> &&minmax_computer)
    : _threads_size(num_threads), _minmax_computer(std::move(minmax_computer))
  {
    assert(_threads_size > 0);
    assert(_minmax_computer != nullptr);
  }

  ~RecordMinMax() = default;

  void initialize(const std::string &input_model_path);

  void setDataSetFormat(DataSetFormat format)
  {
    assert(format != DataSetFormat::UNKNOWN); // FIX_CALLER UNLESS
    _data_set_format = format;
  }

  DataSetFormat getDataSetFormat() const { return _data_set_format; }

  void setInputDataPath(const std::string &input_data_path) { _input_data_path = input_data_path; }

  void profileData();

  void profileDataInParallel(const std::string &input_data_path);

  void saveModel(const std::string &output_model_path);

private:
  luci_interpreter::Interpreter *getInterpreter() const { return _interpreters[0].get(); }

  // Never return nullptr
  MinMaxObserver *getObserver() const
  {
    assert(_observers.size() > 0); // FIX CALLER UNLESS
    assert(_observers[0].get());   // FIX CALLER UNLESS
    return _observers[0].get();
  }

  std::unique_ptr<DataSetIterator> createIterator();

  WholeOutput importH5Data(const std::string &input_data_path);

  std::unique_ptr<luci::Module> _module;

  // Multiple interpreters are used for parallel execution
  std::vector<std::unique_ptr<luci_interpreter::Interpreter>> _interpreters;
  std::vector<std::unique_ptr<MinMaxObserver>> _observers;

  uint32_t _threads_size = 0;
  std::unique_ptr<MinMaxComputer> _minmax_computer;

  DataSetFormat _data_set_format = UNKNOWN;
  std::string _input_data_path;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_H__
