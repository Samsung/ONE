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

#include "moco/Pass/Passes/ResolveFusedBatchNorm.h"

#include <moco/Support/NodeAs.h>

#include <moco/IR/Nodes/TFAdd.h>
#include <moco/IR/Nodes/TFConst.h>
#include <moco/IR/Nodes/TFMul.h>
#include <moco/IR/Nodes/TFFusedBatchNorm.h>

#include <cassert>
#include <cmath>
#include <memory>

namespace
{

bool is_same_shape(moco::TFConst *lc, moco::TFConst *rc)
{
  if (lc->rank() != rc->rank())
    return false;

  for (auto r = 0; r < lc->rank(); ++r)
  {
    if (lc->dim(r).value() != rc->dim(r).value())
      return false;
  }
  return true;
}

void copy_shape(const moco::TFConst *src, moco::TFConst *dst)
{
  assert(src != nullptr);
  assert(dst != nullptr);

  uint32_t rank = src->rank();
  dst->rank(rank);
  for (uint32_t index = 0; index < rank; ++index)
  {
    if (src->dim(index).known())
      dst->dim(index) = src->dim(index).value();
    else
      dst->dim(index).unset();
  }
}

/**
 * @note resolve_to_muladd() will transform TFFusedBatchNorm to TFMul, TFAdd and two ConstGen
 *
 * <arguments>
 * %0:input
 * %1:gamma    : const
 * %2:beta     : const
 * %3:mean     : const
 * %4:variance : const
 * %5:epsilon  : const
 *
 * <constant operations>
 * fbn_epsilon_array = make_array(%5:epsilon)
 * fbn_epsilon = %4:variance + fbn_epsilon_array
 * fbn_rsqrt = 1.0 / math::sqrt(fbn_epsilon)
 *
 * fbn_mean = %3:mean
 * fbn_mul = fbn_rsqrt * %1:gamma
 * fbn_offset = %2:beta
 *
 * fbn_mul_0_param = fbn_mul
 * fbn_add_param = fbn_offset - fbn_mean * fbn_mul
 *
 * <new replace nodes>
 * %11:fbn_mul_0_param = ConstGen(fbn_mul_0_param)
 * %12:fbn_mul_0 = TFMul(%0:input, %11:fbn_mul_0_param)
 * %21:fbn_add_param = ConstGen(fbn_add_param)
 * %22:fbn = TFAdd(%12:fbn_mul_0,%21:fbn_add_param)
 */
bool resolve_to_muladd(loco::Graph *graph, moco::TFFusedBatchNorm *node)
{
  // LOGGER(lfbn);

  auto tffbn_x = node->x();
  if (tffbn_x == nullptr)
  {
    // This node is already converted
    return false;
  }

  auto tffbn_scale = dynamic_cast<moco::TFConst *>(node->scale());
  auto tffbn_offset = dynamic_cast<moco::TFConst *>(node->offset());
  auto tffbn_mean = dynamic_cast<moco::TFConst *>(node->mean());
  auto tffbn_variance = dynamic_cast<moco::TFConst *>(node->variance());

  // all should be const
  if (tffbn_scale == nullptr || tffbn_offset == nullptr || tffbn_mean == nullptr ||
      tffbn_variance == nullptr)
  {
    // INFO(lfbn) << "TFFBN resolve_to_muladd: One of constant input node is not a constant"
    //            << std::endl;
    return false;
  }
  assert(tffbn_scale->dtype() == loco::DataType::FLOAT32);
  assert(tffbn_offset->dtype() == loco::DataType::FLOAT32);
  assert(tffbn_mean->dtype() == loco::DataType::FLOAT32);
  assert(tffbn_variance->dtype() == loco::DataType::FLOAT32);

  // check all const shape are the same
  if (!is_same_shape(tffbn_scale, tffbn_offset) || !is_same_shape(tffbn_scale, tffbn_mean) ||
      !is_same_shape(tffbn_scale, tffbn_variance))
  {
    // INFO(lfbn) << "TFFBN resolve_to_muladd: Shape of constant are not same" << std::endl;
    return false;
  }

  auto tffbn_epsilon = node->epsilon();
  // INFO(lfbn) << "TFFBN tffbn_epsilon = " << tffbn_epsilon << std::endl;
  auto const_num_elements = tffbn_scale->size<loco::DataType::FLOAT32>();
  // INFO(lfbn) << "TFFBN const_num_elements = " << const_num_elements << std::endl;

  // fbn_epsilon = %4:variance + fbn_epsilon_array
  std::unique_ptr<float[]> fbn_epsilon{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    auto variance = tffbn_variance->at<loco::DataType::FLOAT32>(i);
    fbn_epsilon.get()[i] = variance + tffbn_epsilon;
  }

  // fbn_rsqrt = 1.0 / math::sqrt(fbn_epsilon)
  std::unique_ptr<float[]> fbn_rsqrt{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    fbn_rsqrt.get()[i] = 1.0 / sqrt(fbn_epsilon.get()[i]);
  }

  // fbn_mean = %3:mean : TODO remove this block and use %3:mean
  std::unique_ptr<float[]> fbn_mean{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    fbn_mean.get()[i] = tffbn_mean->at<loco::DataType::FLOAT32>(i);
  }

  // fbn_mul = fbn_rsqrt * %1:gamma
  std::unique_ptr<float[]> fbn_mul{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    fbn_mul.get()[i] = fbn_rsqrt.get()[i] * tffbn_scale->at<loco::DataType::FLOAT32>(i);
  }

  // fbn_offset = %2:beta : TODO remove this block and use %2:beta
  std::unique_ptr<float[]> fbn_offset{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    fbn_offset.get()[i] = tffbn_offset->at<loco::DataType::FLOAT32>(i);
  }

  // fbn_mul_0_param = fbn_mul : remove this and use fbn_mul
  std::unique_ptr<float[]> fbn_mul_0_param{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    fbn_mul_0_param.get()[i] = fbn_mul.get()[i];
  }

  // fbn_add_param = fbn_offset - fbn_mean * fbn_mul
  std::unique_ptr<float[]> fbn_add_param{new float[const_num_elements]};
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    fbn_add_param.get()[i] = fbn_offset.get()[i] - fbn_mean.get()[i] * fbn_mul.get()[i];
  }

  // INFO(lfbn) << "TFFBN create ConstGen" << std::endl;

  /*
   * %11:fbn_mul_0_param = ConstGen(fbn_mul_0_param)
   * %21:fbn_add_param = ConstGen(fbn_add_param)
   */
  auto const_fbn_mul_0_param = graph->nodes()->create<moco::TFConst>();
  const_fbn_mul_0_param->dtype(loco::DataType::FLOAT32);
  copy_shape(tffbn_scale, const_fbn_mul_0_param);
  const_fbn_mul_0_param->size<loco::DataType::FLOAT32>(const_num_elements);
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    const_fbn_mul_0_param->at<loco::DataType::FLOAT32>(i) = fbn_mul_0_param.get()[i];
  }
  auto const_fbn_add_param = graph->nodes()->create<moco::TFConst>();
  const_fbn_add_param->dtype(loco::DataType::FLOAT32);
  copy_shape(tffbn_scale, const_fbn_add_param);
  const_fbn_add_param->size<loco::DataType::FLOAT32>(const_num_elements);
  for (int32_t i = 0; i < const_num_elements; i++)
  {
    const_fbn_add_param->at<loco::DataType::FLOAT32>(i) = fbn_add_param.get()[i];
  }

  // INFO(lfbn) << "TFFBN create TFMul, TFAdd" << std::endl;
  /*
   * %12:fbn_mul_0 = TFMul(%0:input, %11:fbn_mul_0_param)
   * %22:fbn = TFAdd(%12:fbn_mul_0,%21:fbn_add_param)
   */
  auto fbn_mul_0 = graph->nodes()->create<moco::TFMul>();
  fbn_mul_0->x(tffbn_x);
  fbn_mul_0->y(const_fbn_mul_0_param);

  auto fbn = graph->nodes()->create<moco::TFAdd>();
  fbn->x(fbn_mul_0);
  fbn->y(const_fbn_add_param);

  // replace old node with new fbn
  replace(node).with(fbn);
  // unlink from graph
  node->x(nullptr);
  node->scale(nullptr);
  node->offset(nullptr);
  node->mean(nullptr);
  node->variance(nullptr);

  return true;
}

} // namespace

namespace moco
{

bool ResolveFusedBatchNorm::run(loco::Graph *graph)
{
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (as<moco::TFFusedBatchNorm>(node))
    {
      if (resolve_to_muladd(graph, as<moco::TFFusedBatchNorm>(node)))
      {
        // tree has been changed. let's return so that we don't need to
        // considier about following node is correct or not.
        return true;
      }
    }
  }

  return false;
}

} // namespace moco
