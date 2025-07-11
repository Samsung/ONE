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

#ifndef __ONERT_IR_LAYOUT_H__
#define __ONERT_IR_LAYOUT_H__

#include <functional>
#include <stdexcept>
#include <string>

namespace onert::ir
{

enum class Layout
{
  UNKNOWN = 0,
  NHWC,
  NCHW
};

// PermuteType::SAME is used for data forwarding and type conversion
enum class PermuteType
{
  NHWC_TO_NCHW,
  NCHW_TO_NHWC,
  SAME
};

inline std::string to_string(Layout layout)
{
  switch (layout)
  {
    case Layout::NHWC:
      return std::string{"NHWC"};
    case Layout::NCHW:
      return std::string{"NCHW"};
    case Layout::UNKNOWN:
      return std::string{"UNKNOWN"};
    default:
      throw std::runtime_error("WRONG LAYOUT");
  }
}

} // namespace onert::ir

namespace std
{

template <> struct hash<onert::ir::Layout>
{
  size_t operator()(onert::ir::Layout value) const noexcept
  {
    using type = typename std::underlying_type<onert::ir::Layout>::type;
    return hash<type>()(static_cast<type>(value));
  }
};

} // namespace std

#endif // __ONERT_IR_LAYOUT_H__
