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

#include "Optimizations.h"
#include "CodeIndex.h"

#include <cassert>

namespace enco
{

void generate_bypass_shuffle(enco::Code *code)
{
  auto m = code->module();

  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    // NOTE The current implementation assumes that all the updates occurs before the first read
    // TODO Remove this assumption
    for (auto u : coco::updaters(bag))
    {
      if ((u->loc() == nullptr) || (u->loc()->asShuffle() == nullptr))
      {
        // Skip if updater is not a Shuffle instruction
        continue;
      }

      for (auto r : coco::readers(bag))
      {
        if ((r->loc() == nullptr) || (r->loc()->asShuffle() == nullptr))
        {
          // Skip if reader is not a Shuffle instruction
          continue;
        }

        auto shuffle_1 = u->loc()->asShuffle();
        auto shuffle_2 = r->loc()->asShuffle();

        // Construct a shuffle instruction
        auto shuffle_3 = m->entity()->instr()->create<coco::Shuffle>();

        shuffle_3->from(shuffle_1->from());
        shuffle_3->into(shuffle_2->into());

        // Attempt to construct a valid bypass shuffle instruction
        bool valid = true;

        for (const auto &C : shuffle_2->range())
        {
          auto B = shuffle_2->at(C);

          if (!shuffle_1->defined(B))
          {
            valid = false;
            break;
          }

          auto A = shuffle_1->at(B);

          shuffle_3->insert(A, C);
        }

        if (valid)
        {
          // Insert shuffle_3 before shuffle_2 if shuffle_3 is a valid bypass of shuffle_2
          shuffle_3->insertBefore(shuffle_2);

          // NOTE shuffle_2 SHOULD BE detached and destroyed after shuffle_3 is inserted
          shuffle_2->detach();
          m->entity()->instr()->destroy(shuffle_2);
        }
        else
        {
          // Destroy shuffle_3 (bypass shuffle) if it is invalid
          m->entity()->instr()->destroy(shuffle_3);
        }
      }
    }
  }
}

} // namespace enco

//
// Hoist Object
//
namespace
{

bool hoistable(const coco::Shuffle *shuffle)
{
  auto range = shuffle->range();

  if (range.size() != shuffle->into()->size())
  {
    return false;
  }

  for (const auto &dst : range)
  {
    if (shuffle->at(dst).value() != dst.value())
    {
      return false;
    }
  }

  return true;
}

bool complete(const coco::Shuffle *s) { return s->range().size() == s->into()->size(); }

bool compatible(const coco::Shuffle *s1, const coco::Shuffle *s2)
{
  if (s1->from() != s2->from())
  {
    return false;
  }

  if (s1->into()->size() != s2->into()->size())
  {
    return false;
  }

  auto range_1 = s1->range();
  auto range_2 = s2->range();

  if (range_1.size() != range_2.size())
  {
    return false;
  }

  bool res = true;

  for (const auto &dst : range_2)
  {
    if (!s1->defined(dst))
    {
      res = false;
      break;
    }

    auto src_1 = s1->at(dst);
    auto src_2 = s2->at(dst);

    if (src_1.value() != src_2.value())
    {
      res = false;
      break;
    }
  }

  return res;
}

} // namespace

namespace enco
{

void hoist_object(enco::Code *code)
{
  auto m = code->module();

  //
  // Case 1
  //
  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    if (auto shuffle = m->entity()->instr()->at(n)->asShuffle())
    {
      if (shuffle->parent() == nullptr)
      {
        continue;
      }

      if (hoistable(shuffle))
      {
        auto from = shuffle->from();
        auto into = shuffle->into();

        into->replaceAllDepsWith(from);
      }
    }
  }

  //
  // Case 2
  //
  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    std::map<CodeIndex, coco::Shuffle *> collected;

    for (auto reader : coco::readers(bag))
    {
      if (auto ins = reader->loc())
      {
        if (auto shuffle = ins->asShuffle())
        {
          collected[code_index(shuffle)] = shuffle;
        }
      }
    }

    std::vector<coco::Shuffle *> sorted;

    for (auto it = collected.begin(); it != collected.end(); ++it)
    {
      sorted.emplace_back(it->second);
    }

    for (uint32_t curr = 0; curr < sorted.size(); ++curr)
    {
      auto const curr_ins = sorted.at(curr);
      auto const curr_bag = curr_ins->into();

      if (!complete(curr_ins))
      {
        continue;
      }

      for (uint32_t next = curr + 1; next < sorted.size(); ++next)
      {
        auto const next_ins = sorted.at(next);
        auto const next_bag = next_ins->into();

        if (!complete(next_ins))
        {
          continue;
        }

        if (compatible(curr_ins, next_ins))
        {
          next_bag->replaceAllDepsWith(curr_bag);
        }
      }
    }
  }
}

} // namespace enco
