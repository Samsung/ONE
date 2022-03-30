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

#ifndef __CIRCLE_EVAL_DIFF_MODULE_EVAL_DIFF_H__
#define __CIRCLE_EVAL_DIFF_MODULE_EVAL_DIFF_H__

#include "MetricPrinter.h"

#include <luci/IR/Module.h>

#include <memory>

namespace circle_eval_diff
{

class ModuleEvalDiff
{
public:
  ModuleEvalDiff(std::unique_ptr<luci::Module> &&first, std::unique_ptr<luci::Module> &&second,
                 std::unique_ptr<MetricPrinter> &&metric)
    : _first_module(std::move(first)), _second_module(std::move(second)), _metric(std::move(metric))
  {
  }

  virtual ~ModuleEvalDiff() = default;

  // Implement this in the child class
  virtual void evalDiff(const std::string &first_input_data_path,
                        const std::string &second_input_data_path) const = 0;

protected:
  std::unique_ptr<luci::Module> _first_module;
  std::unique_ptr<luci::Module> _second_module;
  std::unique_ptr<MetricPrinter> _metric;
};

class H5InputEvalDiff final : public ModuleEvalDiff
{
public:
  H5InputEvalDiff(std::unique_ptr<luci::Module> &&first, std::unique_ptr<luci::Module> &&second,
                  std::unique_ptr<MetricPrinter> &&metric)
    : ModuleEvalDiff(std::move(first), std::move(second), std::move(metric))
  {
  }

  void evalDiff(const std::string &first_input_data_path,
                const std::string &second_input_data_path) const;
};

// TODO Implement ModuleEvalDiff for random input and directory input

} // namespace circle_eval_diff

#endif // __CIRCLE_EVAL_DIFF_MODULE_EVAL_DIFF_H__
