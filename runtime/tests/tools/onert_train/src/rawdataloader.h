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

#ifndef __ONERT_TRAIN_RAWDATALOADER_H__
#define __ONERT_TRAIN_RAWDATALOADER_H__

#include "dataloader.h"

namespace onert_train
{

class RawDataLoader : public DataLoader
{
public:
  RawDataLoader(const std::string &input_file, const std::string &expected_file,
                const std::vector<nnfw_tensorinfo> &input_infos,
                const std::vector<nnfw_tensorinfo> &expected_infos);

  std::tuple<Generator, uint32_t> loadData(const uint32_t batch_size, const float from = 0.0f,
                                           const float to = 1.0f) override;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_RAWDATALOADER_H__
