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
#include "common.h"
#include <fstream>
#include <stdio.h>
#include <json/json.h>

void build_partition_map()
{
  Json::Value root;
  Json::Value graphs(Json::arrayValue);
  int num = 31;

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

TEST_F(ValidationTestPipelineSession, input_tensorinfo_pipeline)
{
  nnfw_tensorinfo t_input;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(_session, 0, &t_input));

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, output_tensorinfo_pipeline)
{
  nnfw_tensorinfo t_output;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(_session, 0, &t_output));

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, input_size_pipeline)
{
  uint32_t input_num = -1;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  NNFW_ENSURE_SUCCESS(nnfw_input_size(_session, &input_num));

  ASSERT_EQ(input_num, 1);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, output_size_pipeline)
{
  uint32_t output_num = -1;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  NNFW_ENSURE_SUCCESS(nnfw_output_size(_session, &output_num));

  ASSERT_EQ(output_num, 1);
  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, set_input_tensorinfo_pipeline)
{
  nnfw_tensorinfo t_input_original;
  nnfw_tensorinfo t_input_after;
  nnfw_tensorinfo t_input = {NNFW_TYPE_TENSOR_FLOAT32, 4, {1, 224, 224, 3}};

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(_session, 0, &t_input_original));
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 0, &t_input));
  NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(_session, 0, &t_input_after));

  ASSERT_TRUE(tensorInfoEqual(t_input_original, t_input_after));

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, input_output_tensorindex)
{
  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  uint32_t input_index = 100;
  NNFW_ENSURE_SUCCESS(nnfw_input_tensorindex(_session, "input", &input_index));
  ASSERT_EQ(input_index, 0);

  uint32_t output_index = 100;
  NNFW_ENSURE_SUCCESS(
    nnfw_output_tensorindex(_session, "MobilenetV1/Predictions/Reshape_1", &output_index));
  ASSERT_EQ(output_index, 0);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_create_pipeline_001)
{
  ASSERT_EQ(nnfw_create_session(nullptr), NNFW_STATUS_UNEXPECTED_NULL);
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

TEST_F(ValidationTestPipelineSession, neg_set_in_pipeline)
{
  float input_buf[1 * 224 * 224 * 3];

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input_buf, sizeof(input_buf)),
            NNFW_STATUS_ERROR);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_set_out_pipeline)
{
  float output_buf[1 * 1001];

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, output_buf, sizeof(output_buf)),
            NNFW_STATUS_ERROR);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_input_tensorinfo_pipeline_001)
{
  nnfw_tensorinfo t_input;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_input_tensorinfo(nullptr, 0, &t_input), NNFW_STATUS_UNEXPECTED_NULL);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_input_tensorinfo_pipeline_002)
{
  nnfw_tensorinfo t_input;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_input_tensorinfo(_session, 1, &t_input), NNFW_STATUS_ERROR);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_output_tensorinfo_pipeline_001)
{
  nnfw_tensorinfo t_output;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_output_tensorinfo(nullptr, 0, &t_output), NNFW_STATUS_UNEXPECTED_NULL);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_output_tensorinfo_pipeline_002)
{
  nnfw_tensorinfo t_output;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_output_tensorinfo(_session, 1, &t_output), NNFW_STATUS_ERROR);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_input_output_size_pipeline)
{
  uint32_t input_num = -1;
  uint32_t output_num = -1;

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_input_size(nullptr, &input_num), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(input_num, -1);
  ASSERT_EQ(nnfw_output_size(nullptr, &output_num), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(output_num, -1);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_set_input_tensorinfo_pipeline)
{
  nnfw_tensorinfo t_input = {NNFW_TYPE_TENSOR_FLOAT32, 4, {1, 224, 224, 3}};

  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  ASSERT_EQ(nnfw_set_input_tensorinfo(nullptr, 0, &t_input), NNFW_STATUS_UNEXPECTED_NULL);

  remove("./partition_map.json");
}

TEST_F(ValidationTestPipelineSession, neg_input_output_tensorindex)
{
  build_partition_map();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_modelfile(
    _session, NNPackages::get().getModelAbsoluteFilePath("mobilenet_v1_1.0_224").c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_prepare_pipeline(_session, "./partition_map.json"));

  uint32_t input_index = 100;
  ASSERT_EQ(nnfw_input_tensorindex(_session, "input1", &input_index), NNFW_STATUS_ERROR);
  ASSERT_EQ(input_index, 100);

  uint32_t output_index = 100;
  ASSERT_EQ(nnfw_output_tensorindex(_session, "MobilenetV1/Predictions/Reshape_2", &output_index),
            NNFW_STATUS_ERROR);
  ASSERT_EQ(output_index, 100);

  remove("./partition_map.json");
}
