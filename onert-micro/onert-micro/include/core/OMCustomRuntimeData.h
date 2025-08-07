/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef ONERT_MICRO_CORE_CUSTOM_RUNTIME_DATA_H
#define ONERT_MICRO_CORE_CUSTOM_RUNTIME_DATA_H

#include "OMRuntimeData.h"

namespace onert_micro
{
namespace core
{

// clang-format off

// ------------------------------------------------------------------------------------------------

template <typename T>
class OMReduceDataContext : public OMDataContext<T, OMAxisContextMixin<1>>
{
public:
  template <class RuntimeKernel>
  explicit OMReduceDataContext(RuntimeKernel &rt_kernel)
    : OMDataContext<T, OMAxisContextMixin<1>>(rt_kernel)
  {}

  ~OMReduceDataContext() override = default;
};

// ------------------------------------------------------------------------------------------------

} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_CUSTOM_RUNTIME_DATA_H
