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

#ifndef __HERMES_CONFIG_H__
#define __HERMES_CONFIG_H__

#include "hermes/core/Severity.h" // TODO Put this into SourceSetting.h
#include "hermes/core/SourceSetting.h"

namespace hermes
{

// TODO Introduce Source.forward.h
class Source;

/**
 * @brief Top-level configuration interface
 *
 * All Hermes configurations SHOULD inherit this interface.
 */
struct Config
{
  virtual ~Config() = default;

  virtual void configure(const Source *, SourceSetting &) const = 0;
};

} // namespace hermes

#endif // __HERMES_CONFIG_H__
