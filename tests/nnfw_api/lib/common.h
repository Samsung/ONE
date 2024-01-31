/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_API_TEST_COMMON_H__
#define __NNFW_API_TEST_COMMON_H__

#include <gtest/gtest.h>
#include <nnfw.h>

bool tensorInfoEqual(const nnfw_tensorinfo &info1, const nnfw_tensorinfo &info2);
uint64_t tensorInfoNumElements(const nnfw_tensorinfo &info);

#endif // __NNFW_API_TEST_COMMON_H__
