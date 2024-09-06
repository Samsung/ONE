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

#ifndef __NNFW_CKER_TRAIN_TYPES_H__
#define __NNFW_CKER_TRAIN_TYPES_H__

namespace nnfw
{
namespace cker
{
namespace train
{

enum class LossReductionType
{
  SUM_OVER_BATCH_SIZE,
  SUM,
};

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TYPES_H__
