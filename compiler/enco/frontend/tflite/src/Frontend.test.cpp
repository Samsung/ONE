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

#include "Frontend.h"

#include <memory>

#include <gtest/gtest.h>

using std::make_unique;

namespace
{

struct MockRawModel final : public RawModel
{
  const tflite::Model *model(void) const override { return nullptr; }
};

} // namespace

TEST(FrontendTest, constructor)
{
  // Let's test whether Frontend is actually constructible.
  auto frontend = make_unique<Frontend>(make_unique<MockRawModel>());

  ASSERT_NE(frontend, nullptr);
}
