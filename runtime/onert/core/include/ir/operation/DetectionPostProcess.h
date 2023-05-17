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

#ifndef __NEURUN_MODEL_OPERATION_DETECTION_POST_PROCESS_NODE_H__
#define __NEURUN_MODEL_OPERATION_DETECTION_POST_PROCESS_NODE_H__

#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

class DetectionPostProcess : public Operation
{
public:
  enum Input
  {
    BOXES = 0,
    SCORES = 1,
    INPUT_ANCHORS = 2
  };

  enum Output
  {
    BOX_COORDS = 0,
    BOX_CLASSES = 1,
    BOX_SCORES = 2,
    NUM_SELECTED = 3
  };

  struct Scale
  {
    float y_scale;
    float x_scale;
    float h_scale;
    float w_scale;
  };

  struct Param
  {
    int max_detections;
    float score_threshold;
    float iou_threshold; // intersection-over-union
    int max_boxes_per_class;
    int32_t num_classes;
    int32_t max_classes_per_detection;
    // N*N complexity instead of N*N*M, where N - number of boxes and M number of classes
    bool center_size_boxes;
    bool do_fast_eval = true;
    Scale scale;
  };

public:
  DetectionPostProcess(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
                       const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  void accept(MutableOperationVisitor &v) override;

  std::string getName() const { return "DetectionPostProcess"; }

public:
  const Param &param() const { return _param; }
  OpCode opcode() const final { return OpCode::DetectionPostProcess; }

private:
  Param _param;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __NEURUN_MODEL_OPERATION_DETECTION_POST_PROCESS_NODE_H__
