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

#ifndef __ONERT_BACKEND_BACKEND_H__
#define __ONERT_BACKEND_BACKEND_H__

#include <memory>

#include "ir/Graph.h"
#include "backend/IConfig.h"
#include "backend/BackendContext.h"

namespace onert
{
namespace backend
{

namespace custom
{
class IKernelBuilder;
}

class Backend
{
public:
  virtual ~Backend() = default;
  virtual std::shared_ptr<onert::backend::IConfig> config() const = 0;

  virtual std::unique_ptr<BackendContext> newContext(ContextData &&) const = 0;
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BACKEND_H__
