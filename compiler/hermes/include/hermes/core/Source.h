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

#ifndef __HERMES_SOURCE_H__
#define __HERMES_SOURCE_H__

#include "hermes/core/Config.h"
#include "hermes/core/Severity.h"
#include "hermes/core/MessageBus.h"
#include "hermes/core/MessageBuffer.h"
#include "hermes/core/SourceSetting.h"

namespace hermes
{

/**
 * @brief Message Source
 *
 * "Source" is the actual interface for users. "Source" accepts log messages from client.
 */
class Source
{
public:
  struct Registry
  {
    virtual ~Registry() = default;

    // NOTE Each "Source" SHOULD outlive "Registry"
    virtual void attach(Source *) = 0;
    virtual void detach(Source *) = 0;
  };

  // NOTE This using statement is introduced for backward compatibility
  // TODO Remove this using declaration after migration
  using Setting = SourceSetting;

protected:
  Source();
  virtual ~Source();

protected:
  // Each "Source" implementation SHOULD invoke activate/deactivate appropriately
  void activate(Registry *, MessageBus *);
  void deactivate(void);

protected:
  Setting &setting(void) { return _setting; }

public:
  /**
   * @brief Check whether a message with a given severity is acceptable or not
   *
   *
   * NOTE This routine is performance critical as app always invokes this routine
   *      (even when logging is disabled).
   */
  inline bool check(const Severity &s) const
  {
    return static_cast<int32_t>(s.level()) < _setting.limit(s.category()).level();
  }

public:
  /**
   * @brief Update Source with a given configuration
   *
   * WARNING Do NOT invoke this manually.
   *
   * TODO Remove virtual after migration
   */
  virtual void reload(const Config *);

public:
  std::unique_ptr<MessageBuffer> buffer(const Severity &) const;

private:
  Setting _setting;

private:
  Registry *_reg = nullptr;
  MessageBus *_bus = nullptr;
};

} // namespace hermes

#define HERMES_FATAL(s)             \
  if ((s).check(::hermes::fatal())) \
  (s).buffer(::hermes::fatal())->os()

#define HERMES_ERROR(s)             \
  if ((s).check(::hermes::error())) \
  (s).buffer(::hermes::error())->os()

#define HERMES_WARN(s)             \
  if ((s).check(::hermes::warn())) \
  (s).buffer(::hermes::warn())->os()

#define HERMES_INFO(s)             \
  if ((s).check(::hermes::info())) \
  (s).buffer(::hermes::info())->os()

#define HERMES_VERBOSE(s, lv)             \
  if ((s).check(::hermes::verbose((lv)))) \
  (s).buffer(::hermes::verbose((lv)))->os()

#endif // __HERMES_SOURCE_H__
