/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Pass/Passes/FuseBinaryIntoPreceding.h"

#include <moco/Support/TFShapeInferenceHelper.h>

#include <moco/IR/TFDialect.h>
#include <moco/IR/Nodes/TFAdd.h>
#include <moco/IR/Nodes/TFBiasAdd.h>
#include <moco/IR/Nodes/TFConst.h>
#include <moco/IR/Nodes/TFConv2D.h>
#include <moco/IR/Nodes/TFDepthwiseConv2dNative.h>
#include <moco/IR/Nodes/TFMul.h>

#include <cassert>
#include <memory>

namespace
{

/**
 * @brief Fusable operation type
 */
enum class FuseType
{
  Conv2D,
  DepthwiseConv2D,
  // TODO Support FullyConnected
};

// TODO rename this method when there is a better name
bool is_only_one_valid(moco::TFConst *xc, moco::TFConst *yc)
{
  if (xc == nullptr && yc == nullptr)
    return false;
  if (xc != nullptr && yc != nullptr)
    return false;

  return true;
}

// TODO Put this in some common place
void copy_shape(const moco::TFConst *src, moco::TFConst *dst)
{
  assert(src != nullptr);
  assert(dst != nullptr);

  uint32_t rank = src->rank();
  dst->rank(rank);
  for (uint32_t index = 0; index < rank; ++index)
  {
    if (src->dim(index).known())
      dst->dim(index) = src->dim(index);
    else
      dst->dim(index).unset();
  }
}

/**
 * @brief return true if shape is identical
 */
bool shape_match(const moco::TFConst *c1, const moco::TFConst *c2)
{
  assert(c1 != nullptr);
  assert(c2 != nullptr);

  uint32_t rank = c1->rank();
  if (rank != c2->rank())
    return false;

  for (uint32_t index = 0; index < rank; ++index)
  {
    if (!c1->dim(index).known() || !c2->dim(index).known())
      return false;

    if (c1->dim(index).value() != c2->dim(index).value())
      return false;
  }
  return true;
}

template <FuseType FT>
moco::TFConst *create_kernel_from_fuse_mulparam(loco::Graph *graph, moco::TFConst *ker,
                                                moco::TFConst *mulparam);

template <>
moco::TFConst *create_kernel_from_fuse_mulparam<FuseType::Conv2D>(loco::Graph *graph,
                                                                  moco::TFConst *ker,
                                                                  moco::TFConst *mulparam)
{
  auto ker_shape_inf = moco::node_shape(ker);
  assert(ker_shape_inf.domain() != loco::Domain::Unknown);
  auto ker_shape = ker_shape_inf.as<loco::TensorShape>();

  auto mulparam_shape_inf = moco::node_shape(mulparam);
  assert(mulparam_shape_inf.domain() != loco::Domain::Unknown);
  auto mulparam_shape = mulparam_shape_inf.as<loco::TensorShape>();

  // create new ker_fused with same size of ker
  auto ker_fused = graph->nodes()->create<moco::TFConst>();

  assert(ker_shape.rank() == 4);
  assert(mulparam_shape.rank() == 1);
  assert(ker_shape.dim(3).value() == mulparam_shape.dim(0).value());

  ker_fused->dtype(loco::DataType::FLOAT32);
  copy_shape(ker, ker_fused);
  auto ker_num_elements = ker->size<loco::DataType::FLOAT32>();
  ker_fused->size<loco::DataType::FLOAT32>(ker_num_elements);

  // TensorFlow Conv2D Kernel has HWIO format
  // Broadcast Mul vector to Kernel tensor by the Output
  const uint32_t ker_height = ker_shape.dim(0).value();
  const uint32_t ker_width = ker_shape.dim(1).value();
  const uint32_t ker_input = ker_shape.dim(2).value();
  const uint32_t ker_output = ker_shape.dim(3).value();

  for (uint32_t ker_y = 0; ker_y < ker_height; ++ker_y)
  {
    for (uint32_t ker_x = 0; ker_x < ker_width; ++ker_x)
    {
      for (uint32_t in_ch = 0; in_ch < ker_input; ++in_ch)
      {
        uint32_t num_items = ((ker_y * ker_width + ker_x) * ker_input + in_ch) * ker_output;
        for (uint32_t out_ch = 0; out_ch < ker_output; ++out_ch)
        {
          auto mulparam_v = mulparam->at<loco::DataType::FLOAT32>(out_ch);
          auto ker_v = ker->at<loco::DataType::FLOAT32>(num_items + out_ch);
          ker_fused->at<loco::DataType::FLOAT32>(num_items + out_ch) = ker_v * mulparam_v;
        }
      }
    }
  }

  return ker_fused;
}

/**
 * @brief Create a kernel from fuse mulparam<FuseType::DepthwiseConv2D> object
 * @return Kernel of fused mulparam
 */
template <>
moco::TFConst *create_kernel_from_fuse_mulparam<FuseType::DepthwiseConv2D>(loco::Graph *graph,
                                                                           moco::TFConst *ker,
                                                                           moco::TFConst *mulparam)
{
  auto ker_shape_inf = moco::node_shape(ker);
  assert(ker_shape_inf.domain() != loco::Domain::Unknown);
  auto ker_shape = ker_shape_inf.as<loco::TensorShape>();

  auto mulparam_shape_inf = moco::node_shape(mulparam);
  assert(mulparam_shape_inf.domain() != loco::Domain::Unknown);
  auto mulparam_shape = mulparam_shape_inf.as<loco::TensorShape>();

  // create new ker_fused with same size of ker
  auto ker_fused = graph->nodes()->create<moco::TFConst>();

  assert(ker_shape.rank() == 4);
  assert(mulparam_shape.rank() == 1);
  assert(ker_shape.dim(2).value() * ker_shape.dim(3).value() == mulparam_shape.dim(0).value());

  ker_fused->dtype(loco::DataType::FLOAT32);
  copy_shape(ker, ker_fused);
  auto ker_num_elements = ker->size<loco::DataType::FLOAT32>();
  ker_fused->size<loco::DataType::FLOAT32>(ker_num_elements);

  // TensorFlow DepthwiseConv2DNative Kernel has HWIM format
  // Broadcast Mul vector to Kernel tensor by the Output
  const uint32_t ker_height = ker_shape.dim(0).value();
  const uint32_t ker_width = ker_shape.dim(1).value();
  const uint32_t ker_input = ker_shape.dim(2).value();
  const uint32_t ker_multiplier = ker_shape.dim(3).value();

  for (uint32_t ker_y = 0; ker_y < ker_height; ++ker_y)
  {
    for (uint32_t ker_x = 0; ker_x < ker_width; ++ker_x)
    {
      for (uint32_t in_ch = 0; in_ch < ker_input; ++in_ch)
      {
        uint32_t num_items = ((ker_y * ker_width + ker_x) * ker_input + in_ch) * ker_multiplier;
        for (uint32_t ker_ch = 0; ker_ch < ker_multiplier; ++ker_ch)
        {
          auto mulparam_v = mulparam->at<loco::DataType::FLOAT32>(in_ch + ker_ch * ker_input);
          auto ker_v = ker->at<loco::DataType::FLOAT32>(num_items + ker_ch);
          ker_fused->at<loco::DataType::FLOAT32>(num_items + ker_ch) = ker_v * mulparam_v;
        }
      }
    }
  }

  return ker_fused;
}

/**
 * @brief Create a fused convolution opertion from kernel of fused mulparam
 * @return Fused convolution operation
 */
template <FuseType FT, class T>
T *fused_conv_node(loco::Graph *graph, moco::TFConst *mulparam, T *conv_node)
{
  // LOGGER(l);

  // ker should be constant
  auto ker = dynamic_cast<moco::TFConst *>(conv_node->filter());
  if (ker == nullptr)
  {
    // Wait until ker is becomes TFConst: there are cases when it's Identity.
    // INFO(l) << "Mul fuse_to_preceding: precedingOp ker is not TFConst";
    return nullptr;
  }
  auto ifm = conv_node->input();
  assert(ifm != nullptr);

  // we need shape information, if not wait till it's ready
  auto ker_shape_inf = moco::node_shape(ker);
  if (ker_shape_inf.domain() == loco::Domain::Unknown)
  {
    // INFO(l) << "Mul fuse_to_preceding: precedingOp ker has no shape";
    return nullptr;
  }
  auto mulparam_shape_inf = moco::node_shape(mulparam);
  if (mulparam_shape_inf.domain() == loco::Domain::Unknown)
  {
    // INFO(l) << "Mul fuse_to_preceding: precedingOp mulparam has no shape";
    return nullptr;
  }
  // if MulParam rank is not 1 we cannot fuse, just skip
  auto mulparam_shape = mulparam_shape_inf.as<loco::TensorShape>();
  if (mulparam_shape.rank() != 1)
  {
    // INFO(l) << "Mul fuse_to_preceding: Mul rank is not 1";
    return nullptr;
  }

  auto ker_fused = create_kernel_from_fuse_mulparam<FT>(graph, ker, mulparam);
  auto conv_fused = graph->nodes()->create<T>();

  conv_fused->input(ifm);
  conv_fused->filter(ker_fused);
  conv_fused->padding(conv_node->padding());
  conv_fused->data_layout(conv_node->data_layout());
  conv_fused->strides(conv_node->strides());

  return conv_fused;
}

/**
 * @note This creates fused ker:2 from ker:1, 'mulparam' and
 *       new precedingOp:2 that uses ker:2 as the kernel.
 *       Then make C to use precedingOp:2 as new input.
 *
 * <Before>
 *                                   mulparam-\
 *          ker:1  --\                         \
 *          ifm  ----- precedingOp:1 ----------- Mul --- C
 *
 *
 * <After>
 *                                   mulparam-\
 *          ker:1  --\                         \
 *                   - precedingOp:1 ----------- Mul ---
 *                   /
 *          ifm  ----- precedingOp:2 ------------------- C
 *          ker:2 ---/
 *
 *
 * [Where]
 *     - precedingOp:1 can be one of TFConv2D, TFDepthwiseConv2dNative, FullyConnected
 *     - 'mulparam' and Mul will be disconnected from the Output.
 *     - ker:2 is added with fused values of ker:1 and mulparam
 *     - precedingOp:2 is added using ifm and ker:2 and other parameters
 *       same as precedingOp:1.
 *     - ker:1, precedingOp:1, 'mulparam' and Mul should be removed in
 *       RemoveDeadNodeTransform if not used.
 */
bool fuse_to_preceding(loco::Graph *graph, moco::TFMul *node)
{
  auto xc = dynamic_cast<moco::TFConst *>(node->x());
  auto yc = dynamic_cast<moco::TFConst *>(node->y());

  // Note: if both are constants, it should be done by constant-folding
  if (!(is_only_one_valid(xc, yc)))
    return false;

  moco::TFConst *mulparam = nullptr;
  moco::TFNode *precedingOp = nullptr;

  if (xc != nullptr)
  {
    mulparam = xc;
    precedingOp = dynamic_cast<moco::TFNode *>(node->y());
  }
  else // yc != nullptr
  {
    mulparam = yc;
    precedingOp = dynamic_cast<moco::TFNode *>(node->x());
  }

  assert(mulparam->dtype() == loco::DataType::FLOAT32);

  // TODO support FullyConnected
  moco::TFNode *fused_node = nullptr;
  if (auto conv2d = dynamic_cast<moco::TFConv2D *>(precedingOp))
    fused_node = fused_conv_node<FuseType::Conv2D, moco::TFConv2D>(graph, mulparam, conv2d);
  else if (auto dw_conv2d = dynamic_cast<moco::TFDepthwiseConv2dNative *>(precedingOp))
    fused_node = fused_conv_node<FuseType::DepthwiseConv2D, moco::TFDepthwiseConv2dNative>(
      graph, mulparam, dw_conv2d);

  // Not ready yet
  if (fused_node == nullptr)
    return false;

  // Replace TFMul node with new precedingOp with fused kernel
  // This will leave existing precedingOp as-is but can be removed if not used
  // from other transformations
  replace(node).with(fused_node);
  // TODO check if need to disconnect
  // node->x(nullptr);
  // node->y(nullptr);
  // fused_node->ifm(nullptr);
  // fused_node->ker(nullptr);

  return true;
}

/**
 * @brief Create zero-filled BiasAdd opertion and insert after precedingOp
 *        The plan is to fuse 'addparam' to TFBiasAdd bias
 * @return Zero-filled BiasAdd operation
 */
template <class T>
moco::TFBiasAdd *create_biasadd_node(loco::Graph *graph, moco::TFConst *addparam, T *precedingOp)
{
  auto dtype = addparam->dtype();
  assert(dtype == loco::DataType::FLOAT32);

  // Create TFConst(bias of TFBiasAdd) with same shape and dtype of 'addparam' but
  // with values 0.0
  auto biasadd_param = graph->nodes()->create<moco::TFConst>();
  biasadd_param->dtype(dtype);
  copy_shape(addparam, biasadd_param);
  auto biasadd_num_elements = addparam->size<loco::DataType::FLOAT32>();
  biasadd_param->size<loco::DataType::FLOAT32>(biasadd_num_elements);
  for (int32_t i = 0; i < biasadd_num_elements; i++)
  {
    biasadd_param->at<loco::DataType::FLOAT32>(i) = 0.0f;
  }

  // Create TFBiasAdd with same shape as TFAdd
  auto data_layout = precedingOp->data_layout();
  auto tf_biasadd = graph->nodes()->create<moco::TFBiasAdd>();
  tf_biasadd->data_layout(data_layout);

  loco::replace(precedingOp).with(tf_biasadd);
  tf_biasadd->value(precedingOp);
  tf_biasadd->bias(biasadd_param);

  return tf_biasadd;
}

/**
 * @note TFAdd will be fused to TFBiasAdd
 *
 * <Before>
 * If precedingOp is not TFBiasAdd, then insert TFConst:1 + TFBiasAdd that
 * TFConst:1 has zero values.
 *
 *                                      addparam --\
 *                                                  \
 *          precedingOp  ---------------------------- TFAdd ----- C
 *
 *
 * <Intermediate>
 * If it's TFBiasAdd and one of the input is TFConst type,
 * then we can fuse 'addparam' to the input TFConst:2 value of TFBiasAdd, where
 * TFConst:2 has added values from 'addparam'
 *
 *                                      addparam --\
 *          TFConst:1  --------\                    \
 *          precedingOp  ------- TFBiasAdd ---------- TFAdd ----- C
 *
 *
 * <After>
 *                                      addparam --\
 *          TFConst:2  --------\                    \
 *          precedingOp  ------- TFBiasAdd ---------- TFAdd -----
 *                                         \--------------------- C
 *
 *
 * [Where]
 *     - precedingOp can be TFConv2D, TFDepthwiseConv2dNative, FullyConnected,
 *       TFBiasAdd.
 *     - Intermediate is to insert TFBiasAdd + TFConst:1
 *     - After is to fuse 'addparam' of TFAdd into TFConst:1 + TFBiasAdd
 *       that becomes TFConst:2 + TFBiasAdd
 */
bool fuse_to_preceding(loco::Graph *graph, moco::TFAdd *node)
{
  // LOGGER(l);

  auto xc = dynamic_cast<moco::TFConst *>(node->x());
  auto yc = dynamic_cast<moco::TFConst *>(node->y());

  // Note: if both are constants, it should be done by constant-folding
  if (!(is_only_one_valid(xc, yc)))
    return false;

  moco::TFConst *addparam = nullptr;
  moco::TFNode *precedingOp = nullptr;

  if (xc != nullptr)
  {
    addparam = xc;
    precedingOp = dynamic_cast<moco::TFNode *>(node->y());
  }
  else // yc != nullptr
  {
    addparam = yc;
    precedingOp = dynamic_cast<moco::TFNode *>(node->x());
  }

  auto addparam_shape_inf = moco::node_shape(addparam);
  if (addparam_shape_inf.domain() == loco::Domain::Unknown)
  {
    // INFO(l) << "Add fuse_to_preceding: addparam has no shape";
    return false;
  }
  // if AddParam rank is not 0 or 1 we cannot fuse, just skip
  auto addparam_shape = addparam_shape_inf.as<loco::TensorShape>();
  if (addparam_shape.rank() > 1)
  {
    // INFO(l) << "Add fuse_to_preceding: Add rank is not 0 or 1";
    return false;
  }

  // TODO do something when rank() is 0
  if (addparam_shape.rank() == 0)
  {
    // Not supported yet
    return false;
  }
  assert(addparam_shape.rank() != 0);

  // TODO support FullyConnected
  moco::TFBiasAdd *biasadd = nullptr;
  if (auto conv2d = dynamic_cast<moco::TFConv2D *>(precedingOp))
    biasadd = create_biasadd_node<moco::TFConv2D>(graph, addparam, conv2d);
  else if (auto dw_conv2d = dynamic_cast<moco::TFDepthwiseConv2dNative *>(precedingOp))
    biasadd = create_biasadd_node<moco::TFDepthwiseConv2dNative>(graph, addparam, dw_conv2d);
  else if (auto old_bias_add = dynamic_cast<moco::TFBiasAdd *>(precedingOp))
    biasadd = old_bias_add;

  if (biasadd == nullptr)
  {
    // try next turn
    return false;
  }

  // Let's fuse addparam into biasadd bias
  auto biasadd_bias = loco::must_cast<moco::TFConst *>(biasadd->bias());
  if (!shape_match(biasadd_bias, addparam))
  {
    // INFO(l) << "TFBiasAdd bias and TFAdd input shape mismatch";
    return false;
  }
  auto add_num_elements = addparam->size<loco::DataType::FLOAT32>();
  assert(add_num_elements == biasadd_bias->size<loco::DataType::FLOAT32>());
  for (int32_t i = 0; i < add_num_elements; i++)
  {
    biasadd_bias->at<loco::DataType::FLOAT32>(i) += addparam->at<loco::DataType::FLOAT32>(i);
  }

  replace(node).with(biasadd);
  // TODO check if need to disconnect
  // node->x(nullptr);
  // node->y(nullptr);

  return true;
}

} // namespace

namespace moco
{

bool FuseBinaryIntoPreceding::run(loco::Graph *graph)
{
  bool changed = false;
  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));

  for (auto node : active_nodes)
  {
    if (node->dialect() == moco::TFDialect::get())
    {
      {
        auto tf_node = dynamic_cast<moco::TFMul *>(node);
        if (tf_node != nullptr)
        {
          if (fuse_to_preceding(graph, tf_node))
            changed = true;
        }
      }
      {
        // TODO support Div
      }

      {
        auto tf_node = dynamic_cast<moco::TFAdd *>(node);
        if (tf_node != nullptr)
        {
          if (fuse_to_preceding(graph, tf_node))
            changed = true;
        }
      }
      {
        // TODO support Sub
      }
    }
  }

  return changed;
}

} // namespace moco
