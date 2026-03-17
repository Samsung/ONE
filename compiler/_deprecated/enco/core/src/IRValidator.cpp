/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "IRValidator.h"

#include <cassert>

namespace enco
{

coco::FeatureShape output_shape(coco::Conv2D *conv2D)
{
  auto load = conv2D->arg()->asLoad();
  assert(load);

  auto ifm = load->object()->asFeature();
  assert(ifm);

  auto ker = conv2D->ker();
  auto stride = conv2D->stride();
  auto pad = conv2D->pad();

  auto striding_width = ifm->shape().width() + pad->left() + pad->right() - ker->shape().width();
  auto striding_height = ifm->shape().height() + pad->top() + pad->bottom() - ker->shape().height();

  // Normally the formula is round(striding_width)/stride->horizontal.
  // in coco IR, striding_width should be a multiple of stride->horizontal(), so round(...) was
  // removed. So does striding_height.
  assert(striding_width % stride->horizontal() == 0);
  assert(striding_height % stride->vertical() == 0);

  auto ofm_width = striding_width / stride->horizontal() + 1;
  auto ofm_height = striding_height / stride->vertical() + 1;

  return coco::FeatureShape(ifm->shape().batch(), ker->shape().count(), ofm_height, ofm_width);
}

bool validate_output_shape(Code *code)
{
  auto module = code->module();

  // for each eval ( conv2d ( ... ) ), check the output shape of conv2D matches output of eval
  for (auto blk = module->block()->head(); blk; blk = blk->next())
  {
    for (auto instr = blk->instr()->head(); instr; instr = instr->next())
    {
      auto eval = instr->asEval();
      if (eval == nullptr)
        continue;

      auto op = eval->op();
      if (!op->asConv2D())
        continue;

      auto conv2D = op->asConv2D();
      auto expected_shape = output_shape(conv2D);

      auto eval_out = eval->out()->asFeature();
      assert(eval_out);

      auto actual_shape = eval_out->shape();

      if (actual_shape != expected_shape)
        return false;
    }
  }
  return true;
}

bool validate(Code *code) { return validate_output_shape(code); }

} // namespace enco
