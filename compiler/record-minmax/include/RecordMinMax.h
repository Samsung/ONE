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

#include <memory>

namespace record_minmax
{

class RecordMinMax
{
public:
  explicit RecordMinMax() = default;

  ~RecordMinMax() = default;

  void initialize(const std::string &input_model_path);

  void profileData(const std::string &mode, const std::string &input_data_path,
                   float min_percentile, float max_percentile);

  void profileRawData(const std::string &mode, const std::string &input_data_path,
                      float min_percentile, float max_percentile);

  void profileRawDataDirectory(const std::string &mode, const std::string &input_data_path,
                               float min_percentile, float max_percentile);

  void profileDataWithRandomInputs(const std::string &mode, float min_percentile,
                                   float max_percentile);

  void saveModel(const std::string &output_model_path);

private:
  std::unique_ptr<luci::Module> _module;
  std::unique_ptr<luci_interpreter::Interpreter> _interpreter;
  std::unique_ptr<MinMaxObserver> _observer;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_H__
