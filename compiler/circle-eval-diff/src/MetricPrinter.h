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

#ifndef __CIRCLE_EVAL_DIFF_METRIC_PRINTER_H__
#define __CIRCLE_EVAL_DIFF_METRIC_PRINTER_H__

#include <luci/IR/Module.h>

#include "Tensor.h"

#include <vector>
#include <iostream>

namespace circle_eval_diff
{

class MetricPrinter
{
public:
  virtual ~MetricPrinter() = default;

  // Child class can implement this function if necessary
  // NOTE init can be skipped
  virtual void init(const luci::Module *, const luci::Module *) {}

  // Accumulate results of comparing the first and the second model's outputs
  virtual void accumulate(const std::vector<std::shared_ptr<Tensor>> &first,
                          const std::vector<std::shared_ptr<Tensor>> &second) = 0;

  // Dump the final result of the corresponding metric
  virtual void dump(std::ostream &os) const = 0;
};

static inline std::ostream &operator<<(std::ostream &os, const MetricPrinter *m)
{
  m->dump(os);
  return os;
}

// Mean Absolute Error
class MAEPrinter final : public MetricPrinter
{
public:
  void init(const luci::Module *first, const luci::Module *second);

  void accumulate(const std::vector<std::shared_ptr<Tensor>> &first,
                  const std::vector<std::shared_ptr<Tensor>> &second);

  void dump(std::ostream &os) const;

private:
  void accum_absolute_error(uint32_t index, const std::shared_ptr<Tensor> a,
                            const std::shared_ptr<Tensor> b);

private:
  // Store accumulated sum of absolute error for each output
  std::vector<Tensor> _intermediate;
  std::vector<std::string> _output_names;
  uint32_t _num_data = 0;
};

} // namespace circle_eval_diff

#endif // __CIRCLE_EVAL_DIFF_METRIC_PRINTER_H__
