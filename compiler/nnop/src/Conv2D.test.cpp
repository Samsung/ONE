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

#include "nnop/Conv2D.h"

#include <nncc/core/ADT/feature/Overlay.h>
#include <nncc/core/ADT/feature/CHWLayout.h>

#include <nncc/core/ADT/kernel/Overlay.h>
#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <gtest/gtest.h>

using namespace nnop;
using namespace nncc::core::ADT;

TEST(CONV2D, conv_1)
{
  const feature::Shape ofm_shape{1, 1, 1};
  int ofm_data[] = {0};
  auto ofm_overlay = feature::make_overlay<int, feature::CHWLayout>(ofm_shape, ofm_data);

  const feature::Shape ifm_shape{1, 3, 3};
  int ifm_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto ifm_overlay = feature::make_overlay<int, feature::CHWLayout>(ifm_shape, ifm_data);

  const kernel::Shape ker_shape{1, 1, 3, 3};
  int ker_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto ker_overlay = kernel::make_overlay<int, kernel::NCHWLayout>(ker_shape, ker_data);

  const PadInfo pad{0, 0, 0, 0};
  const StrideInfo stride{1, 1};

  nnop::conv(ofm_shape, ofm_overlay, ifm_shape, ifm_overlay, ker_shape, ker_overlay, pad, stride);

  EXPECT_EQ(ofm_data[0], 204);
}
