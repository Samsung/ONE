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

#ifndef __ONERT_TRAIN_DATALOADER_H__
#define __ONERT_TRAIN_DATALOADER_H__

#include "allocation.h"
#include "nnfw.h"

#include <functional>
#include <vector>
#include <tuple>
#include <fstream>

namespace onert_train
{

using Generator = std::function<bool(uint32_t,                  /** index **/
                                     std::vector<Allocation> &, /** input **/
                                     std::vector<Allocation> & /** expected **/)>;

class DataLoader
{
public:
  DataLoader(const std::vector<nnfw_tensorinfo> &input_infos,
             const std::vector<nnfw_tensorinfo> &expected_infos)
    : _input_infos{input_infos}, _expected_infos{expected_infos}
  {
    // DO NOTHING
  }
  virtual ~DataLoader() = default;

  virtual std::tuple<Generator, uint32_t>
  loadData(const uint32_t batch_size, const float from = 0.0f, const float to = 1.0f) = 0;

protected:
  std::vector<nnfw_tensorinfo> _input_infos;
  std::vector<nnfw_tensorinfo> _expected_infos;
  std::ifstream _input_file;
  std::ifstream _expected_file;
  uint32_t _data_length;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_DATALOADER_H__
