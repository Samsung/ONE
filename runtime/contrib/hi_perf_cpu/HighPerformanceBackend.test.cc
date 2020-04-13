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

#include <gtest/gtest.h>

#include "nnpack.h"

TEST(High_performance_backend, NNPACK_Test)
{
  // Check that it is possible to import
  const enum nnp_status init_status = nnp_initialize();

  // One of the allowed nnp status codes
  ASSERT_GE(init_status, 0);
  ASSERT_LE(init_status, 54);

  // If it is possible to test, test relu
  if (init_status == nnp_status_success)
  {
    float in[] = {-1, 1, -1, 1};
    float out[4];
    nnp_relu_output(1, 4, in, out, 0, nullptr);
    for (int i = 0; i < 4; i++)
    {
      ASSERT_EQ(out[i], in[i] >= 0 ? in[i] : 0);
    }
  }
  nnp_deinitialize();
}
