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

#include "CopyLowering.h"

#include <set>
#include <cassert>

//
// Lower Copy as Shuffle
//
namespace enco
{

void lower_copy(enco::Code *code)
{
  auto m = code->module();

  std::set<coco::Copy *> lowered_copies;

  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    auto ins = m->entity()->instr()->at(n);

    assert(ins != nullptr);

    if (ins->parent() == nullptr)
    {
      // Skip if instruction does not belong to a list
      continue;
    }

    auto copy = ins->asCopy();

    if (copy == nullptr)
    {
      // Skip if instruction is not a copy
      continue;
    }

    // TODO Support non-Feature objects
    auto ifm = copy->from()->asFeature();
    auto ofm = copy->into()->asFeature();

    if ((ifm == nullptr) || (ofm == nullptr))
    {
      continue;
    }

    assert(ifm->layout()->batch() == ofm->layout()->batch());
    assert(ifm->layout()->shape() == ofm->layout()->shape());

    auto shuffle = m->entity()->instr()->create<coco::Shuffle>();

    shuffle->from(ifm->bag());
    shuffle->into(ofm->bag());

    const uint32_t B = ifm->layout()->batch();
    const uint32_t C = ifm->layout()->shape().depth();
    const uint32_t H = ifm->layout()->shape().height();
    const uint32_t W = ifm->layout()->shape().width();

    for (uint32_t b = 0; b < B; ++b)
    {
      for (uint32_t ch = 0; ch < C; ++ch)
      {
        for (uint32_t row = 0; row < H; ++row)
        {
          for (uint32_t col = 0; col < W; ++col)
          {
            const auto from = ifm->layout()->at(b, ch, row, col);
            const auto into = ofm->layout()->at(b, ch, row, col);

            shuffle->insert(from, into);
          }
        }
      }
    }

    shuffle->insertBefore(copy);
    lowered_copies.insert(copy);
  }

  // Destroy lowered copy
  for (const auto &copy : lowered_copies)
  {
    copy->detach();
    m->entity()->instr()->destroy(copy);
  }
}

} // namespace enco
