/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_MISC_TENSOR_DIFF_SUMMARY_H__
#define __NNFW_MISC_TENSOR_DIFF_SUMMARY_H__

#include "Zipper.h"
#include "Comparator.h"

#include "misc/fp32.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{

class DiffSummary : public nnfw::misc::tensor::Comparator::Observer
{
public:
  DiffSummary()
    : max_abs_diff_index(0), max_abs_diff_expected{0.0f}, max_abs_diff_obtained{0.0f},
      max_abs_diff_value{0.0f}, max_rel_diff_index(0), max_rel_diff_expected{0.0f},
      max_rel_diff_obtained{0.0f}, max_rel_diff_value{0.0f}
  {
    // DO NOTHING
  }

public:
  void notify(const nnfw::misc::tensor::Index &index, float expected, float obtained) override;

public:
  nnfw::misc::tensor::Index max_abs_diff_index;
  float max_abs_diff_expected;
  float max_abs_diff_obtained;
  float max_abs_diff_value;

  nnfw::misc::tensor::Index max_rel_diff_index;
  float max_rel_diff_expected;
  float max_rel_diff_obtained;
  float max_rel_diff_value;
};

void DiffSummary::notify(const nnfw::misc::tensor::Index &index, float expected, float obtained)
{
  const auto abs_diff_value = std::fabs(expected - obtained);

  if (max_abs_diff_value < abs_diff_value)
  {
    max_abs_diff_index = index;
    max_abs_diff_value = abs_diff_value;
    max_abs_diff_expected = expected;
    max_abs_diff_obtained = obtained;
  }

  const auto rel_diff_value = nnfw::misc::fp32::relative_diff(expected, obtained);

  if (max_rel_diff_value < rel_diff_value)
  {
    max_rel_diff_index = index;
    max_rel_diff_value = rel_diff_value;
    max_rel_diff_expected = expected;
    max_rel_diff_obtained = obtained;
  }
}

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_DIFF_SUMMARY_H__
