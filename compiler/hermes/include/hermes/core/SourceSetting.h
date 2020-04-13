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

#ifndef __HERMES_SOURCE_SETTING_H__
#define __HERMES_SOURCE_SETTING_H__

#include <array>
#include <cstdint>

namespace hermes
{

class Filter final
{
public:
  Filter(int32_t *ptr) : _ptr{ptr}
  {
    // DO NOTHING
  }

public:
  inline void reject_all(void) { *_ptr = -1; }
  inline void accept_upto(uint16_t lv) { *_ptr = static_cast<int32_t>(lv); }
  inline void accept_all(void) { *_ptr = 65536; }

private:
  int32_t *_ptr;
};

class Limit final
{
public:
  Limit(const int32_t *ptr) : _ptr{ptr}
  {
    // DO NOTHING
  }

public:
  inline int32_t level(void) const { return *_ptr; }

private:
  const int32_t *_ptr;
};

class SourceSetting final
{
public:
  SourceSetting()
  {
    // Reject all the messages by default
    reject_all();
  }

public:
  void reject_all(void)
  {
    filter(FATAL).reject_all();
    filter(ERROR).reject_all();
    filter(WARN).reject_all();
    filter(INFO).reject_all();
    filter(VERBOSE).reject_all();
  }

  void accept_all(void)
  {
    filter(FATAL).accept_all();
    filter(ERROR).accept_all();
    filter(WARN).accept_all();
    filter(INFO).accept_all();
    filter(VERBOSE).accept_all();
  }

  inline Filter filter(const SeverityCategory &cat)
  {
    return _ulimits.data() + static_cast<uint32_t>(cat);
  }

  inline Limit limit(const SeverityCategory &cat) const
  {
    return _ulimits.data() + static_cast<uint32_t>(cat);
  }

private:
  /**
   * @brief Allowed message level for each category
   *
   * This source will accept all the messages whose level belongs to [0, ulimit)
   *  where ulimit corresdpons to "limit(cat).value()"
   */
  std::array<int32_t, 5> _ulimits;
};

} // namespace hermes

#endif // __HERMES_SOURCE_SETTING_H__
