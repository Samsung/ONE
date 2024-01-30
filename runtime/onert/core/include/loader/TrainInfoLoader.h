/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_LOADER_TRAININFO_LOADER_H__
#define __ONERT_LOADER_TRAININFO_LOADER_H__

#include "ir/train/TrainingInfo.h"
#include "ir/Model.h"

namespace onert
{
namespace train
{
namespace traininfo_loader
{

// TODO change this line to use inline variable after C++17
extern const char *const TRAININFO_METADATA_NAME;

std::unique_ptr<ir::train::TrainingInfo> loadTrainingInfo(const uint8_t *buffer, const size_t size);

} // namespace traininfo_loader
} // namespace train
} // namespace onert

#endif // __ONERT_LOADER_TRAININFO_LOADER_H__
