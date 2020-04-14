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

#include "fixtures.h"

TEST_F(ValidationTestOneOpModelLoaded, prepare_001)
{
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTest, neg_prepare_001) { ASSERT_EQ(nnfw_prepare(nullptr), NNFW_STATUS_ERROR); }
