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

#include "model_path.h"
#include "fixtures.h"

TEST_F(ValidationTestSessionCreated, load_session_001)
{
  // Existing model must
  ASSERT_EQ(nnfw_load_model_from_file(
                _session, ModelPath::get().getModelAbsolutePath(MODEL_ONE_OP_IN_TFLITE).c_str()),
            NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestSessionCreated, neg_load_session_001)
{
  ASSERT_EQ(nnfw_load_model_from_file(
                _session, ModelPath::get().getModelAbsolutePath("nonexisting_directory").c_str()),
            NNFW_STATUS_ERROR);
}
