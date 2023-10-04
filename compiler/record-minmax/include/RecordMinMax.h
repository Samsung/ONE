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
  explicit RecordMinMax(uint32_t num_threads, std::unique_ptr<MinMaxComputer> &&minmax_computer)
    : _threads_size(num_threads), _minmax_computer(std::move(minmax_computer))
  {
    assert(_threads_size > 0);
    assert(_minmax_computer != nullptr);
  }

  ~RecordMinMax() = default;

  void initialize(const std::string &input_model_path);

  // TODO Refactor profile functions
  void profileData(const std::string &input_data_path);

  void profileDataInParallel(const std::string &input_data_path);

  void profileRawData(const std::string &input_data_path);

  void profileRawDataDirectory(const std::string &input_data_path);

  void profileDataWithRandomInputs(void);

  void saveModel(const std::string &output_model_path);

  void loadMinMax(const std::string &output_model_path);

private:
  luci_interpreter::Interpreter *getInterpreter() const { return _interpreters[0].get(); }

  // Never return nullptr
  MinMaxObserver *getObserver() const
  {
    assert(_observers.size() > 0); // FIX CALLER UNLESS
    assert(_observers[0].get());   // FIX CALLER UNLESS
    return _observers[0].get();
  }

  WholeOutput importH5Data(const std::string &input_data_path);

  std::unique_ptr<luci::Module> _module;

  // Multiple interpreters are used for parallel execution
  std::vector<std::unique_ptr<luci_interpreter::Interpreter>> _interpreters;
  std::vector<std::unique_ptr<MinMaxObserver>> _observers;

  uint32_t _threads_size = 0;
  std::unique_ptr<MinMaxComputer> _minmax_computer;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_H__
