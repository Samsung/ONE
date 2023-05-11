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

#include "DetectionPostProcessLayer.h"

#include "ndarray/Array.h"

#include <numeric>
#include <utility>
#include <cmath>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

namespace
{

using namespace ndarray;

using CenterSizeBox = DetectionPostProcessLayer::CenterSizeBox;
using CornerBox = DetectionPostProcessLayer::CornerBox;

using NonMaxSuppressionParam = DetectionPostProcessLayer::DetectionPostProcessParameters;
using Allocations = DetectionPostProcessLayer::Allocations;

struct OutputArrays
{
  OutputArrays(CornerBox *coords_buf, float *scores_buf, float *classes_buf,
               int *num_selections_buf, size_t max_detections)
    : coords(coords_buf, {max_detections}), scores(scores_buf, {max_detections}),
      classes(classes_buf, {max_detections}), num_selections(num_selections_buf, {1})
  {
  }

  Array<CornerBox> coords;
  Array<float> scores;
  Array<float> classes;
  Array<int> num_selections;
};

struct TemporaryArrays
{
  TemporaryArrays(int *selections_buffer, int max_detections)
    : selections(selections_buffer, {static_cast<unsigned long>(max_detections)})
  {
  }

  Array<int> selections;
};

// sort indices in decreasing order of first `k` scores
void PartialArgSort(const ContiguousSpan<float, true> &scores,
                    const ContiguousSpan<int, false> &indices, int k)
{
  std::iota(indices.begin(), indices.begin() + k, 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.begin() + scores.size(),
                    [&scores](const int i, const int j) { return scores[i] > scores[j]; });
}

template <typename T> ContiguousSpan<T, false> static vecToSpan(std::vector<T> &v)
{
  return ContiguousSpan<T, false>{v.begin(), v.end()};
}

Array<const CornerBox> decodeBoxes(const Array<float> &raw_boxes, const Array<float> &raw_anchors,
                                   bool center_box_format, const CenterSizeBox &scales)
{
  auto nbatches = raw_boxes.shape().dim(0);
  auto num_boxes = raw_boxes.shape().dim(1);

  auto anchors = array_cast<const CenterSizeBox>(raw_anchors, {num_boxes});

  if (!center_box_format)
  {
    auto boxes_p = reinterpret_cast<const CornerBox *>(raw_boxes.flat().data());
    return {boxes_p, {num_boxes}};
  }
  else
  {
    // TODO support box center-width encoding correctly
    // i.e anchors
    auto boxes_p = reinterpret_cast<const CenterSizeBox *>(raw_boxes.flat().data());
    Array<const CenterSizeBox> in_boxes{boxes_p, {num_boxes}};

    auto decoded_boxes_p = new CornerBox[nbatches * num_boxes];
    Array<CornerBox> decoded_boxes_a{decoded_boxes_p, {num_boxes}};

    for (size_t i = 0; i < num_boxes; ++i)
    {
      auto anchor = anchors.at(i);
      auto &box = decoded_boxes_a.at(i);
      float yc = in_boxes.at(i).y / scales.y * anchor.h + anchor.y;
      float xc = in_boxes.at(i).x / scales.x * anchor.w + anchor.x;
      float halfh = 0.5f * std::exp(in_boxes.at(i).h / scales.h) * anchor.h;
      float halfw = 0.5f * std::exp(in_boxes.at(i).w / scales.w) * anchor.w;
      box.x1 = xc - halfw;
      box.x2 = xc + halfw;
      box.y1 = yc - halfh;
      box.y2 = yc + halfh;

      assert(box.x2 > box.x1);
      assert(box.y2 > box.y1);
    }

    auto decoded_boxes_a_shape = decoded_boxes_a.shape();

    return array_cast<const CornerBox>(std::move(decoded_boxes_a), decoded_boxes_a_shape);
  }
}

float computeIOU(const CornerBox &box1, const CornerBox &box2)
{
  float area_i = (box1.y2 - box1.y1) * (box1.x2 - box1.x1);
  float area_j = (box2.y2 - box2.y1) * (box2.x2 - box2.x1);
  if (area_i <= 0 || area_j <= 0)
  {
    return 0.0;
  }
  float in_ymin = std::max<float>(box1.y1, box2.y1);
  float in_xmin = std::max<float>(box1.x1, box2.x1);
  float in_ymax = std::min<float>(box1.y2, box2.y2);
  float in_xmax = std::min<float>(box1.x2, box2.x2);
  float in_area = std::max<float>(in_ymax - in_ymin, 0.0) * std::max<float>(in_xmax - in_xmin, 0.0);

  return in_area / (area_i + area_j - in_area);
}

int doSingleClass(const Array<const CornerBox> &boxes, const std::vector<float> &scores,
                  const NonMaxSuppressionParam &param, TemporaryArrays &temps,
                  size_t max_detections)
{
  auto num_boxes = boxes.shape().dim(0);

  std::vector<int> sorted_box_indices(num_boxes);
  PartialArgSort(ContiguousSpan<float, true>(scores.data(), num_boxes),
                 vecToSpan(sorted_box_indices), num_boxes);

  // TODO move to temp allocations
  std::vector<int> process_box(num_boxes, 1);

  size_t selected_count = 0;
  for (size_t i = 0; i < num_boxes; ++i)
  {
    auto box_index = sorted_box_indices[i];

    if (!process_box[box_index] || scores[box_index] < param.score_threshold)
    {
      continue;
    }

    temps.selections.at(selected_count) = box_index;
    selected_count++;

    if (selected_count >= max_detections)
    {
      break;
    }

    for (size_t j = i + 1; j < num_boxes; ++j)
    {
      if (!process_box[sorted_box_indices[j]])
      {
        continue;
      }

      float IOU = computeIOU(boxes.at(box_index), boxes.at(sorted_box_indices[j]));
      if (IOU > param.iou_threshold)
      {
        process_box[sorted_box_indices[j]] = 0;
      }
    }
  }

  return selected_count;
}

void collectBoxes(TemporaryArrays &temporary, const Array<const CornerBox> &decoded_boxes,
                  std::vector<float> &scores, int num_selected, OutputArrays &output,
                  const Array<int> &sorted_classes, int detections_per_box)
{
  auto &selections = temporary.selections;

  size_t output_box_count = 0;

  for (int i = 0; i < num_selected; ++i)
  {
    int selected_box = selections.at(output_box_count);

    for (int c = 0; c < detections_per_box; ++c)
    {
      output.classes.at(output_box_count) = sorted_classes.at(selected_box, c);
      output.scores.at(output_box_count) = scores[selected_box];
      output.coords.at(output_box_count) = decoded_boxes.at(selected_box);
      output_box_count++;
    }
  }
}

void DetectionPostProcess(const Array<float> &boxes_a, const Array<float> &scores_a,
                          Array<float> &num_selected_a, const NonMaxSuppressionParam &param,
                          const Allocations &allocations, OutputArrays &outputs)
{
  TemporaryArrays temporary(allocations.selections_buffer, param.max_detections);

  // Only batch of 1 is supported atm
  auto num_boxes = boxes_a.shape().dim(1);
  size_t num_classes = param.num_classes;
  size_t num_classes_with_background = scores_a.shape().dim(2);
  bool have_background = num_classes_with_background != num_classes;

  size_t max_classes_per_box = std::min<size_t>(num_classes, param.max_classes_per_detection);

  // TODO move this to allocations
  std::vector<int> sorted_class_indices(num_boxes * num_classes);

  Array<int> class_indices(sorted_class_indices.data(), {num_boxes, num_classes});

  // TODO move to allocations
  std::vector<float> max_scores(num_boxes);

  for (size_t row = 0; row < num_boxes; row++)
  {
    auto box_scores = scores_a.slice(0, row).offset(have_background ? 1 : 0);
    auto indices = class_indices.slice(row);

    PartialArgSort(box_scores, indices, num_classes);

    max_scores[row] = box_scores[indices[0]];
  }

  auto anchors_a =
    Array<float>(reinterpret_cast<float *>(param.anchors_input->buffer()), {num_boxes, 4});
  auto decoded_boxes = decodeBoxes(boxes_a, anchors_a, param.center_box_format, param.scales);

  int num_selected =
    doSingleClass(decoded_boxes, max_scores, param, temporary, param.max_detections);

  collectBoxes(temporary, decoded_boxes, max_scores, num_selected, outputs, class_indices,
               max_classes_per_box);

  num_selected_a.at(0) = num_selected;
}
} // namespace

template <typename T> Array<T> toArray(uint8_t *ptr, std::vector<int32_t> &descr)
{
  ndarray::Shape shape(descr.size());
  for (size_t i = 0; i < descr.size(); ++i)
  {
    shape.dim(i) = descr[i];
  }

  return Array<T>{reinterpret_cast<T *>(ptr), shape};
}

void DetectionPostProcessLayer::configure(DetectionPostProcessParameters parameters)
{
  _parameters = std::move(parameters);
  _allocations.selections_buffer = new int[_parameters.max_detections * 2];
}

void DetectionPostProcessLayer::run()
{
  auto nbatches = (unsigned int)_parameters.boxes_descr[0];
  // no suport for batch other than 1( it's fine since tflite does not support
  // batch for postprocess either )
  assert(nbatches == 1);

  auto boxes_a = toArray<float>(_parameters.boxes_input->buffer(), _parameters.boxes_descr);
  auto scores_a = toArray<float>(_parameters.scores_input->buffer(), _parameters.scrores_descr);

  auto num_selected_a = ndarray::Array<float>(
    reinterpret_cast<float *>(_parameters.num_selections_output->buffer()), {nbatches});

  OutputArrays outputArrays(reinterpret_cast<CornerBox *>(_parameters.box_coords_output->buffer()),
                            reinterpret_cast<float *>(_parameters.box_scores_output->buffer()),
                            reinterpret_cast<float *>(_parameters.box_classes_output->buffer()),
                            reinterpret_cast<int *>(_parameters.num_selections_output->buffer()),
                            _parameters.max_detections);

  DetectionPostProcess(boxes_a, scores_a, num_selected_a, _parameters, _allocations, outputArrays);
}

DetectionPostProcessLayer::~DetectionPostProcessLayer() { delete[] _allocations.selections_buffer; }

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
