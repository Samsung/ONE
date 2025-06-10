/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ExecTime.h"

#include <algorithm>
#include <cassert>

namespace onert::exec
{

int64_t ExecTime::getOperationExecTime(const backend::Backend *backend,
                                       const std::string &operation, bool quant,
                                       uint32_t op_size) const
{
  auto found_backend = _measurements.find(backend);
  if (found_backend == _measurements.end())
    return NOT_FOUND; // no execution time for this backend

  auto found_operation_with_type = found_backend->second.find(operation);
  if (found_operation_with_type == found_backend->second.end())
    // no execution time for this operation
    return NOT_FOUND;

  auto found_operation = found_operation_with_type->second.find(quant);
  if (found_operation == found_operation_with_type->second.end())
    // no execution time for this operation
    return NOT_FOUND;

  auto found_size = found_operation->second.find(op_size);
  if (found_size != found_operation->second.end())
    return found_size->second; // found execution time

  // Try to interpolate
  if (found_operation->second.size() < 2)
    // not possible to do linear interpolation
    return found_operation->second.begin()->second;

  // if we reach here, then this means, that there is no record, that is equal to op_size
  auto upper_bound = found_operation->second.upper_bound(op_size); // > op_size
  auto lower_bound = upper_bound;

  if (upper_bound == found_operation->second.end()) // all values <= op_size
  {
    upper_bound--;
    lower_bound = upper_bound;
    lower_bound--;
  }
  else if (upper_bound == found_operation->second.begin()) // all values > op_size
  {
    upper_bound++;
  }
  else // op_size between
  {
    lower_bound--;
  }

  // Linear interpolation
  const auto x0 = static_cast<int64_t>(lower_bound->first); // size
  const auto x1 = static_cast<int64_t>(upper_bound->first); // size
  const int64_t y0 = lower_bound->second;                   // time
  const int64_t y1 = upper_bound->second;                   // time
  const auto x = static_cast<int64_t>(op_size);

  int64_t interpolated_value = y0 + (x - x0) * (y1 - y0) / (x1 - x0);

  // In some cases ops with smaller inputs is executed slower than the one
  // with larger inputs, more likely because of a backend's load difference
  if (interpolated_value < 0 && x > x1)
  {
    return y0;
  }
  // It must be non-positive ONLY if it's lesser than both of them
  assert(interpolated_value > 0 || x < x0);

  // execution time must be non-negative
  return std::max<int64_t>(interpolated_value, 1);
}

void ExecTime::updateOperationExecTime(const backend::Backend *backend,
                                       const std::string &operation, bool quant, uint32_t op_size,
                                       int64_t time)
{
  // If the op is not implemented for some input, it should not be scheduled
  const auto &recs = _measurements[backend][operation][quant];
  if (time == getMax() ||
      std::any_of(recs.begin(), recs.end(),
                  [](std::pair<const uint32_t, const int64_t> p) { return p.second == getMax(); }))
  {
    _measurements[backend][operation][quant].clear();
    _measurements[backend][operation][quant].emplace(op_size, getMax());
  }
  else
  {
    auto it = _measurements[backend][operation][quant].emplace(op_size, time);
    if (!it.second)
    {
      // affect of the last measurement is bigger than the previous ones:
      //   this prefers new metrics than older once, so will adapt backend changes
      it.first->second = (it.first->second + time) / 2;
    }
  }
}

void ExecTime::updatePermuteTime(const backend::Backend *from_backend,
                                 const backend::Backend *to_backend, bool quant, uint32_t op_size,
                                 int64_t time)
{
  updateOperationExecTime(from_backend, to_backend->config()->id(), quant, op_size, time);
}

int64_t ExecTime::getPermuteTime(const backend::Backend *from_backend,
                                 const backend::Backend *to_backend, bool quant,
                                 uint32_t op_size) const
{
  return getOperationExecTime(from_backend, to_backend->config()->id(), quant, op_size);
}

} // namespace onert::exec
