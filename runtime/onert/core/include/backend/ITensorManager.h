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

#ifndef __ONERT_BACKEND_ITENSOR_MANAGER_H__
#define __ONERT_BACKEND_ITENSOR_MANAGER_H__

namespace onert
{
namespace backend
{

// NOTE This name ITensorManager has been discussed whether or not the name is proper.
// Anyone can argue with any better name.
/**
 * @brief Interface as an abstract tensor manager which has MemoryManager
 */
struct ITensorManager
{
  virtual ~ITensorManager() = default;
};

} // namespace backend
} // namespace onert

#include <unordered_set>
#include <memory>

namespace onert
{
namespace backend
{

using TensorManagerSet = std::unordered_set<std::unique_ptr<backend::ITensorManager>>;

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ITENSOR_MANAGER_H__
