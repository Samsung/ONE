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

#ifndef __ONERT_TRAIN_METRICS_H__
#define __ONERT_TRAIN_METRICS_H__

#include "allocation.h"
#include "nnfw.h"
#include <vector>

namespace onert_train
{
class Metrics
{
public:
  Metrics(const std::vector<Allocation> &output, const std::vector<Allocation> &expected,
          const std::vector<nnfw_tensorinfo> &infos);

private:
  template <typename T>
  float categoricalAccuracy(const T *output, const T *expected, uint32_t batch, uint64_t size);

public:
  float categoricalAccuracy(int32_t index);

private:
  const std::vector<Allocation> &_output;
  const std::vector<Allocation> &_expected;
  const std::vector<nnfw_tensorinfo> &_infos;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_METRICS_H__
