/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __HERMES_SINK_H__
#define __HERMES_SINK_H__

#include "hermes/core/Message.h"

#include <memory>

namespace hermes
{

/**
 * @brief Message consumer interface
 *
 * All message consumers should inherit this interface.
 */
struct Sink
{
  struct Registry
  {
    virtual ~Registry() = default;

    // NOTE SinkRegistry takes the ownership of all the appended Sink objects
    virtual void append(std::unique_ptr<Sink> &&) = 0;
  };

  virtual ~Sink() = default;

  virtual void notify(const Message *) = 0;
};

} // namespace hermes

#endif // __HERMES_SINK_H__
