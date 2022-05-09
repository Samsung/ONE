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

#include "InputDataLoader.h"
#include "MetricPrinter.h"

#include <string>
#include <memory>
#include <vector>

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

class CircleEvalDiff final
{
public:
  struct Context
  {
    std::string first_model_path;
    std::string second_model_path;
    std::vector<Metric> metric;
  };

public:
  CircleEvalDiff(std::unique_ptr<Context> &&ctx);

  ~CircleEvalDiff();

  void init();

  // Evaluate two circle models for the given input data and compare the results
  void evalDiff(const InputDataLoader *first, const InputDataLoader *second) const;

public:
  const std::vector<loco::Node *> first_module_inputs(void) const;
  const std::vector<loco::Node *> second_module_inputs(void) const;

private:
  std::unique_ptr<Context> _ctx;
  std::unique_ptr<luci::Module> _first_module;
  std::unique_ptr<luci::Module> _second_module;
  std::vector<std::unique_ptr<MetricPrinter>> _metrics;
};

} // namespace circle_eval_diff

#endif // __CIRCLE_EVAL_DIFF_H__
