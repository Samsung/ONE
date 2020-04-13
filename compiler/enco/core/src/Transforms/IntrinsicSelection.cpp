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

#include "IntrinsicSelection.h"

#include "coex/IR.h"

namespace
{

/**
 * @brief Return a backend-speicific coco (extend) instruction
 *
 * @note rewrite(ins) returns nullptr if selection fails
 */
coco::Instr *rewrite(coco::Instr *curr)
{
  auto m = curr->module();
  assert(m != nullptr);

  if (auto eval = coco::safe_cast<coco::Eval>(curr))
  {
    if (auto concat_f = eval->op()->asConcatF())
    {
      auto fst_load = concat_f->left()->asLoad();
      auto snd_load = concat_f->right()->asLoad();

      if (fst_load && snd_load && (concat_f->axis() == coco::ConcatF::Axis::Depth))
      {
        // Here is the pattern of interest
        //
        //   %ofm = eval(ConcatF(Depth, Load(%left), Load(%right)))
        //
        auto fst_feature = fst_load->object()->asFeature();
        auto snd_feature = snd_load->object()->asFeature();
        assert((fst_feature != nullptr) && (snd_feature != nullptr));

        auto out_feature = eval->out()->asFeature();
        assert(out_feature != nullptr);

        eval->out(nullptr);

        auto depth_concat = m->entity()->instr()->create<ANNDepthConcatF>();

        depth_concat->out(out_feature);
        depth_concat->fst(fst_feature);
        depth_concat->snd(snd_feature);

        return depth_concat;
      }

      return nullptr;
    }
  }

  return nullptr;
}

} // namespace

namespace enco
{

void select_intrinsic(enco::Code *code)
{
  auto m = code->module();

  for (auto blk = m->block()->head(); blk; blk = blk->next())
  {
    auto ins = blk->instr()->head();

    while (ins)
    {
      if (auto rewritten_ins = rewrite(ins))
      {
        rewritten_ins->insertBefore(ins);
        ins->detach();

        ins = rewritten_ins;
      }

      ins = ins->next();
    }
  }
}

} // namespace enco
