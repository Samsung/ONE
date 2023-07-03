/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <backend/basic/train/TrainableTensor.h>

#include <util/Utils.h>

#include <functional>

namespace onert
{
namespace backend
{
namespace basic
{
namespace train
{

std::vector<ITensor *> TrainableTensor::optVars()
{
  std::vector<ITensor *> ret;
  for (auto &&e : _opt_vars)
  {
    ret.emplace_back(e.get());
  }
  return ret;
}

void TrainableTensor::fillBuffer(const std::shared_ptr<ir::Data> &data)
{
  auto *buffer = _tensor.buffer();
  assert(buffer);
  assert(total_size() == data->size());
  std::memcpy(buffer, data->base(), data->size());
}

} // namespace train
} // namespace basic
} // namespace backend
} // namespace onert
