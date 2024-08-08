/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __RECORD_HESSIAN_H__
#define __RECORD_HESSIAN_H__

#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>

#include "record-hessian/HessianObserver.h"
#include "record-hessian/HessianComputer.h"

#include <memory>
#include <thread>

namespace record_hessian
{

using Buffer = std::vector<char>;
using Output = std::vector<Buffer>;
using WholeOutput = std::vector<Output>;

class RecordHessian
{
public:
  explicit RecordHessian(uint32_t num_threads, std::unique_ptr<HessianComputer> &&hessian_computer)
    : _hessian_computer(std::move(hessian_computer))
  {
    assert(_hessian_computer != nullptr);
  }

  ~RecordHessian() = default;

  void initialize(luci::Module *module);
  // TODO Refactor profile functions
  void profileData(const std::string &input_data_path);

  void profileDataInParallel(const std::string &input_data_path);

  void profileRawData(const std::string &input_data_path);

  void profileRawDataDirectory(const std::string &input_data_path);

  void profileDataWithRandomInputs(void);

private:
  luci_interpreter::Interpreter *getInterpreter() const { return _interpreter.get(); }

  // Never return nullptr
  HessianObserver *getObserver() const { return _observer.get(); }

  luci::Module *_module;

  // Multiple interpreters are used for parallel execution
  std::unique_ptr<luci_interpreter::Interpreter> _interpreter;
  std::unique_ptr<HessianObserver> _observer;

  std::unique_ptr<HessianComputer> _hessian_computer;
};

} // namespace record_hessian

#endif // __RECORD_HESSIAN_H__
