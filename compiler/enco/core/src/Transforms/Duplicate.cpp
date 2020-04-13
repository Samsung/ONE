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

#include "Duplicate.h"

#include <map>
#include <set>

#include <cassert>

namespace
{

coco::Block *find_or_create_first_block(coco::Module *m)
{
  if (m->block()->empty())
  {
    auto blk = m->entity()->block()->create();
    m->block()->append(blk);
    return blk;
  }

  return m->block()->head();
}

} // namespace

namespace
{

class DuplicatePass
{
private:
  void runOnModule(coco::Module *m) const;

public:
  void runOnCode(enco::Code *) const;
};

void DuplicatePass::runOnModule(coco::Module *m) const
{
  // Let's find candidates
  std::set<coco::Bag *> candidates;

  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    if (bag->isInput() && bag->isOutput())
    {
      candidates.insert(bag);
    }
  }

  // Return if there is no candidate
  if (candidates.empty())
  {
    return;
  }

  std::map<const coco::Bag *, coco::Input *> input_map;
  std::map<const coco::Bag *, coco::Output *> output_map;

  for (uint32_t n = 0; n < m->input()->size(); ++n)
  {
    auto input = m->input()->at(n);
    assert(input->bag() != nullptr);
    input_map[input->bag()] = input;
  }

  for (uint32_t n = 0; n < m->output()->size(); ++n)
  {
    auto output = m->output()->at(n);
    assert(output->bag() != nullptr);
    output_map[output->bag()] = output;
  }

  // For each in/out bag,
  //   1. Create a new bag of the same size
  //   2. Copy the content from the original bag
  //   3. Mark the newly created bag as an output
  for (const auto &candidate : candidates)
  {
    assert(coco::updaters(candidate).empty());
    assert(input_map.find(candidate) != input_map.end());
    assert(output_map.find(candidate) != output_map.end());

    auto src = candidate;
    auto dst = m->entity()->bag()->create(src->size());

    // Create a copy instruction
    auto shuffle = m->entity()->instr()->create<coco::Shuffle>();

    shuffle->from(src);
    shuffle->into(dst);

    for (uint32_t n = 0; n < src->size(); ++n)
    {
      shuffle->insert(coco::ElemID{n} /* FROM */, coco::ElemID{n} /* INTO */);
    }

    find_or_create_first_block(m)->instr()->prepend(shuffle);

    // Let's use the new bag as an output
    output_map.at(src)->bag(dst);
  }
}

void DuplicatePass::runOnCode(enco::Code *code) const { runOnModule(code->module()); }

} // namespace

namespace enco
{

void duplicate_inout_bag(enco::Code *code)
{
  DuplicatePass duplicate;
  duplicate.runOnCode(code);
}

} // namespace enco
