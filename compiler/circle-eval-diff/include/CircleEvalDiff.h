/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_EVAL_DIFF_H__
#define __CIRCLE_EVAL_DIFF_H__

#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>

#include <string>
#include <memory>

namespace circle_eval_diff
{

// Forward declaration
class ModuleEvalDiff;

enum class Metric
{
  Undefined, // For debugging
  MAE,       // Mean Absolute Error
  MAPE,      // Mean Percentage Absolute Error
  MPEIR,     // Mean Peak Error to Interval Ratio
};

enum class InputFormat
{
  Undefined, // For debugging
  H5,
  // TODO Implement Random, Directory
};

class CircleEvalDiff final
{
public:
  struct Context
  {
    std::string first_model_path;
    std::string second_model_path;
    std::vector<Metric> metric;
    InputFormat input_format = InputFormat::Undefined;
    std::string output_prefix;
  };

public:
  CircleEvalDiff(std::unique_ptr<Context> &&ctx);

  ~CircleEvalDiff();

  void init();

  // Evaluate two circle models for the given input data and compare the results
  void evalDiff(const std::string &first_input_data_path,
                const std::string &second_input_data_path) const;

private:
  std::unique_ptr<Context> _ctx;
  std::unique_ptr<ModuleEvalDiff> _runner;
};

} // namespace circle_eval_diff

#endif // __CIRCLE_EVAL_DIFF_H__
