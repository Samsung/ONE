/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "GenModelTest.h"

#include <memory>

TEST_F(GenModelTest, OneOp_DetectionPostProcess_SingleBox)
{
  CircleGen cgen;

  int boxes = cgen.addTensor({{1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int scores = cgen.addTensor({{1, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int anchors = cgen.addTensor({{1, 1, 4}, circle::TensorType::TensorType_FLOAT32});

  int box_coors = cgen.addTensor({{1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int box_classes = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int box_scores = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int num_selected = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorDetectionPostProcess(
    {{boxes, scores, anchors}, {box_coors, box_classes, box_scores, num_selected}}, 1, 10, 10, 5, 5,
    0.8, 0.5, 1, 1, 1);
  cgen.setInputsAndOutputs({boxes, scores, anchors},
                           {box_coors, box_classes, box_scores, num_selected});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 0, 0, 0}, {0, 0.9}, {0, 0, 1, 1}},
                                          {{-0.5, -0.5, 0.5, 0.5}, {0}, {0.9}, {1}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DetectionPostProcess_SinglBox_MultiClasses)
{
  CircleGen cgen;

  int boxes = cgen.addTensor({{1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int scores = cgen.addTensor({{1, 1, 3}, circle::TensorType::TensorType_FLOAT32});
  int anchors = cgen.addTensor({{1, 1, 4}, circle::TensorType::TensorType_FLOAT32});

  int box_coors = cgen.addTensor({{1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int box_classes = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int box_scores = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int num_selected = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorDetectionPostProcess(
    {{boxes, scores, anchors}, {box_coors, box_classes, box_scores, num_selected}}, 2, 10, 10, 5, 5,
    0.8, 0.5, 1, 1, 1);
  cgen.setInputsAndOutputs({boxes, scores, anchors},
                           {box_coors, box_classes, box_scores, num_selected});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 0, 0, 0}, {0, 0.7, 0.9}, {0, 0, 1, 1}},
                                          {{-0.5, -0.5, 0.5, 0.5}, {1}, {0.9}, {1}}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
