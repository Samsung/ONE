/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizerLoader.h"

#include <gtest/gtest.h>

using namespace onert::odc;

// Test QuantizerLoader singleton
TEST(odc_QuantizerLoader, singleton)
{
  QuantizerLoader &loader1 = QuantizerLoader::instance();
  QuantizerLoader &loader2 = QuantizerLoader::instance();
  ASSERT_EQ(&loader1, &loader2);
}

// Test load quantizer library
TEST(odc_QuantizerLoader, load)
{
  QuantizerLoader &loader = QuantizerLoader::instance();
  ASSERT_EQ(loader.loadLibrary(), 0);

  // Load twice to check if it is thread-safe
  ASSERT_EQ(loader.loadLibrary(), 0);
}

// Get quantizer function without loading quantizer library
TEST(odc_QuantizerLoader, neg_get)
{
  QuantizerLoader &loader = QuantizerLoader::instance();
  // Unload because it may be loaded on previous tests
  ASSERT_EQ(loader.unloadLibrary(), 0);
  ASSERT_EQ(loader.get(), nullptr);
}

// Check quantizer function pointer when QuantizerLoader is unloaded
TEST(odc_QuantizerLoader, neg_unload)
{
  QuantizerLoader &loader = QuantizerLoader::instance();
  ASSERT_EQ(loader.loadLibrary(), 0);
  ASSERT_EQ(loader.unloadLibrary(), 0);
  ASSERT_EQ(loader.get(), nullptr);
}
