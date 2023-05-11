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

#ifndef __ONERT_BACKEND_TRAINING_OPS_DPP_H__
#define __ONERT_BACKEND_TRAINING_OPS_DPP_H__

#include <exec/IFunction.h>

#include "OperationUtils.h"

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

class DetectionPostProcessLayer : public ::onert::exec::IFunction
{
public:
  struct CornerBox
  {
    float y1, x1;
    float y2, x2;
  };

  struct CenterSizeBox
  {
    float y, x;
    float h, w;
  };

  struct DetectionPostProcessParameters
  {
    const IPortableTensor *boxes_input;
    const IPortableTensor *scores_input;
    const IPortableTensor *anchors_input;
    IPortableTensor *box_coords_output;
    IPortableTensor *box_classes_output;
    IPortableTensor *box_scores_output;
    IPortableTensor *num_selections_output;
    std::vector<int32_t> boxes_descr;
    std::vector<int32_t> scrores_descr;

    uint32_t max_detections;
    float score_threshold;
    float iou_threshold; // intersection-over-union
    uint32_t max_boxes_per_class;
    bool center_box_format = false;
    int32_t num_classes;
    int32_t max_classes_per_detection;
    CenterSizeBox scales;
  };

  enum SelectionFormat
  {
    BOX_INDEX = 1,
    CLASS_INDEX = 0
  };

  struct Allocations
  {
    int *selections_buffer = nullptr;
    // TODO move all dynamic allocations here, and into configure phase
  };

  DetectionPostProcessLayer() : _parameters{}
  {
    // DO NOTHING
  }

  virtual ~DetectionPostProcessLayer();

public:
  void configure(DetectionPostProcessParameters parameters);

  void run() override;

private:
  DetectionPostProcessParameters _parameters;

  Allocations _allocations;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_DPP_H__
