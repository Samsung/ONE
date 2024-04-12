/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_GRAPH_SHAPE_INFERENCE_H__
#define __ONERT_GRAPH_SHAPE_INFERENCE_H__

#include "Utils.h"

#include "ir/operation/Concat.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/operation/Pool2D.h"
#include "ir/operation/Reshape.h"
#include "ir/operation/StridedSlice.h"
#include "compiler/LoweredGraph.h"
#include "ir/Index.h"
#include "ir/Layout.h"
#include "ir/OperationVisitor.h"
#include "backend/ITensor.h"
#include "backend/ITensorRegistry.h"

namespace onert
{
namespace shape_inference
{

using Shapes = std::vector<ir::Shape>;

// Define shape calculation for operations. List them in alphabetic order.

ir::Shape inferArgMinMaxShape(const ir::Shape &input_shape, int axis, int rank);

ir::Shape inferBatchMatMulShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape,
                                const ir::operation::BatchMatMul::Param &param);

ir::Shape inferBCQFullyConnectedShape(const ir::Shape &in_shape, const ir::Shape &cluster_shape,
                                      const int32_t *cluster_buf);

ir::Shape inferBCQGatherShape(const ir::Shape &indices_shape, const ir::Shape &cluster_shape,
                              const int32_t *cluster_buf, int rank,
                              const ir::operation::BCQGather::Param &param);

ir::Shape inferBroadcastToShape(const ir::Shape shp_shape, const int32_t *shp_buf);

ir::Shape inferConcatShape(const Shapes &in_shapes, const ir::operation::Concat::Param &param);

ir::Shape inferConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                           const ir::operation::Conv2D::Param &param,
                           ir::Layout layout = ir::Layout::NHWC);

ir::Shape inferDepthwiseConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                                    const ir::operation::DepthwiseConv2D::Param &param,
                                    ir::Layout layout = ir::Layout::NHWC);

ir::Shape inferEltwiseShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape);

ir::Shape inferExpandDimsShape(const ir::Shape &in_shape, int32_t axis);

template <typename T> ir::Shape inferFillShape(const ir::Shape &fill_shape, const T *shape_buf);

ir::Shape inferFullyConnectedShape(const ir::Shape &in_shape, const ir::Shape &ker_shape);

ir::Shape inferGatherShape(const ir::Shape &input_shape, const ir::Shape &indices_shape, int axis,
                           int rank);

ir::Shape inferOnehotShape(const ir::Shape &input_shape, const int depth, int axis);

ir::Shape inferPackShape(const ir::Shape &input_shape, int axis, int rank, int num);

ir::Shape inferPadShape(const ir::Shape &in_shape, const int32_t *pad_buf, const size_t num_pads);

ir::Shape inferPoolShape(const ir::Shape &in_shape, const ir::operation::Pool2D::Param &param,
                         ir::Layout layout = ir::Layout::NHWC);

template <typename T> ir::Shape inferRangeShape(T start_val, T limit_val, T delta_val);

ir::Shape inferReshapeShape(const ir::Shape &input_shape, const int32_t *shape_buf,
                            const int32_t shape_num_elements);

ir::Shape inferReduceShape(const ir::Shape &input_shape, const std::vector<int> &axes,
                           bool keep_dims);

template <float *> ir::Shape inferRangeShape(float *start_val, float *limit_val, float *delta_val);

template <typename T> ir::Shape inferRangeShape(T start_val, T limit_val, T delta_val);

ir::Shape inferResizeBilinearShape(const ir::Shape &in_shape, const int32_t output_height,
                                   const int32_t output_width);

ir::Shape inferSelectShape(const ir::Shape &input_cond_shape, const ir::Shape &input_true_shape,
                           const ir::Shape &input_false_shape);

template <typename T>
ir::Shape inferSliceShape(const ir::Shape &input_shape, const T *begins_buf, const T *sizes_buf);

ir::Shape inferSpaceToBatchNDShape(const ir::Shape &input_shape, const ir::Shape &block_shape_shape,
                                   const ir::Shape &padding_shape, const int32_t *block_shape_buf,
                                   const int32_t *padding_buf);

ir::Shape inferSplitShape(const ir::Shape input_shape, int axis_value, int num_splits);

ir::Shape inferSqueezeShape(const ir::Shape &in_shape, const ir::operation::Squeeze::Param &param);

struct StridedSliceParams
{
  int8_t start_indices_count;
  int16_t start_indices[4];
  int8_t stop_indices_count;
  int16_t stop_indices[4];
  int8_t strides_count;
  int16_t strides[4];

  int16_t begin_mask;
  int16_t ellipsis_mask;
  int16_t end_mask;
  int16_t new_axis_mask;
  int16_t shrink_axis_mask;
};

template <typename T>
StridedSliceParams buildStridedSliceParams(const T *begin, const T *end, const T *strides,
                                           const uint32_t begin_mask, const uint32_t end_mask,
                                           const uint32_t shrink_axis_mask, const uint8_t rank);

ir::Shape inferStridedSliceShape(const ir::Shape &input_shape, const StridedSliceParams &op_params,
                                 uint32_t rank);

ir::Shape inferTileShape(const ir::Shape &in_shape, const int32_t *multiplier_buf,
                         const int32_t multiplier_size);

ir::Shape inferTransposeShape(const ir::Shape &in_shape, const int32_t *perm_buf,
                              const int32_t rank);

ir::Shape inferUnpackShape(const ir::Shape &input_shape, int axis, int rank);

} // namespace shape_inference
} // namespace onert

#endif // __ONERT_GRAPH_SHAPE_INFERENCE_H__
