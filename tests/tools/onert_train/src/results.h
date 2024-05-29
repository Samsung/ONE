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

#ifndef __ONERT_TRAIN_RESULTS_H__
#define __ONERT_TRAIN_RESULTS_H__

#include <iostream>
#include <vector>

namespace onert_train
{
template <typename T>
class Results
{
public:
  Results(const int num): _losses(num), _metrics(num) {}

  void reset()
  {
    std::fill(_losses.begin(), _losses.end(), 0);
    std::fill(_metrics.begin(), _metrics.end(), 0);
  }

  void setLoss(const int32_t idx, const T var)
  {
    _losses[idx] += var;
  }

  void setMetrics(const int32_t idx, const T var)
  {
    _metrics[idx] += var;
  }

  void printLoss(const int step, const std::string &prefix = std::string{})
  {
    std::streamsize sz = std::cout.precision();
    {
      std::cout << std::setprecision(4) << std::fixed;
      std::cout << " - "<< prefix << "loss: ";
      for (uint32_t i = 0; i < _losses.size(); ++i)
      {
        std::cout << "[" << i << "] " << _losses[i] / step;
      }
    }
    std::cout.precision(sz);
  }

  void printMetrics(const int step, const std::string &prefix = std::string{})
  {
    std::streamsize sz = std::cout.precision();
    {
      std::cout << std::setprecision(4) << std::fixed;
      std::cout << " - " << prefix << ": ";
      for (uint32_t i = 0; i < _metrics.size(); ++i)
      {
        std::cout << "[" << i << "] " << _metrics[i] / step;
      }
    }
    std::cout.precision(sz);
  }

private:
  std::vector<T> _losses;
  std::vector<T> _metrics;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_RESULTS_H__
