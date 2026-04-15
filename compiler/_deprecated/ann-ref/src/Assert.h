/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __ASSERT_H__
#define __ASSERT_H__

#include "Logging.h"

// Assert macro, as Android does not generally support assert.
#define ASSERT(v)                                                                         \
  do                                                                                      \
  {                                                                                       \
    if (!(v))                                                                             \
    {                                                                                     \
      LOG(ERROR) << "'" << #v << "' failed at " << __FILE__ << ":" << __LINE__ << "'\n";  \
      abort();                                                                            \
    }                                                                                     \
  } while (0)

#endif // __ASSERT_H__
