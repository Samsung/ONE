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

#include <caffe_importer.h>
#include <gtest/gtest.h>

#include <exception>
#include <string>

const char *ErrorMsg = "NNC can't load model. Detected problems:\n"
                       "  * DummyData: unsupported layer\n"
                       "  * LSTM: parameter 'expose_hidden' has unsupported value: 1\n"
                       "  * UnexcitingLayerType: unknown layer";

// When adding support for new layers, change the model, not the test
TEST(CAFFE_IMPORT_UNSUPPORTED, ImportAModelWithUnsupportedLayers)
{
  std::string filename = std::string(TEST_DIR) + "unsupported.caffemodel";

  try
  {
    mir_caffe::loadModel(filename);
  }
  catch (std::exception &e)
  {
    ASSERT_EQ(std::string(ErrorMsg), e.what());
    return;
  }

  FAIL();
}
