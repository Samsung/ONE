/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <fstream>
#include <stdio.h>
#include <json/json.h>

void build_partition_map()
{
  Json::Value root;
  Json::Value graphs(Json::arrayValue);
  int num = 31;

  remove("./partition_map.json");

  for (int i = 0; i < num; i++)
  {
    if (i < 7)
      graphs.append(Json::Value(0));
    else
      graphs.append(Json::Value(1));
  }

  root["partition_map"] = graphs;
  root["num_partitions"] = 2;

  Json::StyledWriter sw;
  std::string jsonString = sw.write(root);

  FILE *pFile = NULL;

  pFile = fopen("./partition_map.json", "wt");
  fwrite(jsonString.c_str(), jsonString.length(), 1, pFile);
  fclose(pFile);
}

TEST_F(ValidationTestPipelineSession, create_pipeline_001)
{
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_close_session(_session));
}

TEST_F(ValidationTestPipelineSession, neg_create_pipeline_001)
{
  ASSERT_EQ(nnfw_create_session(nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestPipelineSession, pipeline_session_test_model)
{
  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));
  NNFW_ENSURE_SUCCESS(nnfw_close_session(_session));

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_pipeline_session_model_load)
{
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));

  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  ASSERT_EQ(nnfw_load_model_from_modelfile(
              nullptr, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestPipelineSession, neg_prepare_pipeline_001)
{
  ASSERT_EQ(nnfw_prepare_pipeline(nullptr, nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestPipelineSession, prepare_pipeline_001)
{
  std::ifstream readFile("./partition_map.json");

  if (readFile.good())
  {
    remove("./partition_map.json");
  }

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  ASSERT_EQ(nnfw_prepare_pipeline(_session, "./partition_map.json"), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestPipelineSession, prepare_pipeline_002)
{
  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  remove("./partition_map.json");
}
