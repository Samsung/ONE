/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file     Utils.h
 * @brief    This file contains utility macro
 */

#ifndef __ONERT_UTIL_UTILS_H__
#define __ONERT_UTIL_UTILS_H__

#include <string.h>

#define UNUSED_RELEASE(a) (void)(a)

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/**
 * @brief Check required condition and report
 *        Failed at func:line
 */
#define OP_REQUIRES(EXP)                                                                \
  do                                                                                    \
  {                                                                                     \
    if (!(EXP))                                                                         \
      throw std::runtime_error("Failed at " + std::string(__FILENAME__) + ":" +         \
                               std::string(__func__) + ":" + std::to_string(__LINE__)); \
  } while (0)

/**
 * @brief Check required condition and report
 *        Failed at function:line <error message>
 */
#define OP_REQUIRES_MSG(EXP, MSG)                                                             \
  do                                                                                          \
  {                                                                                           \
    if (!(EXP))                                                                               \
      throw std::runtime_error("Failed at " + std::string(__FILENAME__) + ":" +               \
                               std::string(__func__) + ":" + std::to_string(__LINE__) + " " + \
                               MSG);                                                          \
  } while (0)

#endif // __ONERT_UTIL_UTILS_H__
