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

#include <caffe2_importer.h>
#include <gtest/gtest.h>

#include <exception>
#include <string>

const char *ErrorMsg = "NNC can't load model. Detected problems:\n"
                       "  * Sin: unknown layer";

// When adding support for new layers, change the model, not the test
TEST(CAFFE_IMPORT_UNSUPPORTED, ImportAModelWithUnsupportedLayers)
{

  std::string predict_net = std::string(TEST_DIR) + "/predict_net.pb";
  std::string init_net = std::string(TEST_DIR) + "/init_net.pb";

  try
  {
    mir_caffe2::loadModel(predict_net, init_net, {{1}});
  }
  catch (std::exception &e)
  {
    ASSERT_EQ(std::string(ErrorMsg), e.what());
    return;
  }

  FAIL();
}
