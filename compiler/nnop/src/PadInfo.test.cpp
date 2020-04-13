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

#include "nnop/PadInfo.h"

#include <gtest/gtest.h>

TEST(PAD_INFO, explicit_constructor)
{
  const uint32_t top = 3;
  const uint32_t bottom = 4;
  const uint32_t left = 5;
  const uint32_t right = 6;

  nnop::PadInfo pad_info{top, bottom, left, right};

  ASSERT_EQ(pad_info.top(), top);
  ASSERT_EQ(pad_info.bottom(), bottom);
  ASSERT_EQ(pad_info.left(), left);
  ASSERT_EQ(pad_info.right(), right);
}
