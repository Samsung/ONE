/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __RECORD_MINMAX_MINMAXCOMPUTER_H__
#define __RECORD_MINMAX_MINMAXCOMPUTER_H__

#include "MinMaxObserver.h"

namespace record_minmax
{

class MinMaxComputer
{
public:
  MinMaxComputer()
  {
    // Do nothing
  }

  virtual ~MinMaxComputer() = default;

  // Child class must implement this
  // TODO Use proper input signature
  virtual void update_qparam(MinMaxObserver *observer) = 0;
};

class PercentileComputer : public MinMaxComputer
{
public:
  PercentileComputer(float min_percentile, float max_percentile)
    : _min_percentile(min_percentile), _max_percentile(max_percentile)
  {
  }

  virtual void update_qparam(MinMaxObserver *observer);

private:
  float _min_percentile;
  float _max_percentile;
};

class MovingAvgComputer : public MinMaxComputer
{
public:
  MovingAvgComputer(uint32_t batch_size, float update_const)
    : _batch_size(batch_size), _update_const(update_const)
  {
  }

  virtual void update_qparam(MinMaxObserver *observer);

private:
  uint32_t _batch_size;
  float _update_const;
};

std::unique_ptr<MinMaxComputer> make_percentile_computer(float min_percentile,
                                                         float max_percentile);

std::unique_ptr<MinMaxComputer> make_moving_avg_computer(uint32_t batch_size,
                                                         float moving_avg_const);

} // namespace record_minmax

#endif // __RECORD_MINMAX_MINMAXCOMPUTER_H__
