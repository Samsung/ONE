/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __EVENT_H__
#define __EVENT_H__

#include <NeuralNetworks.h>

#include <memory>

namespace onert
{
namespace exec
{
class Execution;
} // namespace exec
} // namespace onert

struct ANeuralNetworksEvent
{
public:
  ANeuralNetworksEvent(const std::shared_ptr<onert::exec::Execution> &execution);

public:
  bool waitFinish(void) noexcept;

private:
  const std::shared_ptr<onert::exec::Execution> _execution;
};

#endif
