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

#ifndef __DALGONA_H__
#define __DALGONA_H__

#include "OperatorObserver.h"

#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>

#include <memory>

namespace dalgona
{

class Dalgona
{
public:
  explicit Dalgona() = default;

  ~Dalgona() = default;

  void initialize(const std::string &input_model_path);

  void runAnalysis(const std::string &input_data_path, const std::string &analysis_path,
                   const std::string &analysis_args);

private:
  std::unique_ptr<luci::Module> _module{nullptr};
  std::unique_ptr<luci_interpreter::Interpreter> _interpreter{nullptr};
  std::unique_ptr<OperatorObserver> _observer{nullptr};
};

} // namespace dalgona

#endif // __DALGONA_H__
