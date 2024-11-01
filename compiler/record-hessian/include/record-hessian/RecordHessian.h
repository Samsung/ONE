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

namespace record_hessian
{

class RecordHessian
{
public:
  explicit RecordHessian() {}

  ~RecordHessian() = default;

  void initialize(luci::Module *module);
  // TODO Refactor profile functions
  std::unique_ptr<HessianMap> profileData(const std::string &input_data_path);

  std::unique_ptr<HessianMap> profileDataInParallel(const std::string &input_data_path);

  std::unique_ptr<HessianMap> profileRawData(const std::string &input_data_path);

  std::unique_ptr<HessianMap> profileRawDataDirectory(const std::string &input_data_path);

  std::unique_ptr<HessianMap> profileDataWithRandomInputs(void);

private:
  luci_interpreter::Interpreter *getInterpreter() const { return _interpreter.get(); }

  // Never return nullptr
  HessianObserver *getObserver() const { return _observer.get(); }

  luci::Module *_module;

  std::unique_ptr<luci_interpreter::Interpreter> _interpreter;
  std::unique_ptr<HessianObserver> _observer;
};

} // namespace record_hessian

#endif // __RECORD_HESSIAN_H__
