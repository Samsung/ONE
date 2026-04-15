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

#include "DuplicatedObjectReduction.h"

#include "CodeIndex.h"
#include "IRUtils.h"

#include <set>

namespace
{

/**
 * @brief Collect feature objects in coco IR
 */
std::set<coco::FeatureObject *> features(const coco::Module *m)
{
  std::set<coco::FeatureObject *> res;

  for (uint32_t n = 0; n < m->entity()->object()->size(); ++n)
  {
    if (auto feature = m->entity()->object()->at(n)->asFeature())
    {
      res.insert(feature);
    }
  }

  return res;
}

std::set<coco::FeatureObject *> candidates(const coco::FeatureObject *src)
{
  std::set<coco::FeatureObject *> res;

  for (auto consumer : coco::consumers(src))
  {
    if (auto copy = consumer->loc()->asCopy())
    {
      auto dst = copy->into()->asFeature();
      assert(dst != nullptr);

      if (dst->layout()->id() == coco::FeatureLayouts::BHWC::uid())
      {
        res.insert(dst);
      }
    }
  }

  return res;
}

CodeIndex code_index(coco::Object::Producer *p)
{
  if (auto ins = p->loc())
  {
    return ::code_index(ins);
  }

  return CodeIndex{};
}

} // namespace

namespace enco
{

void reduce_duplicated_object(enco::Code *code)
{
  auto m = code->module();

  for (const auto &src : features(m))
  {
    auto copied = candidates(src);

    if (copied.size() <= 1)
    {
      continue;
    }

    // Find the dominator
    coco::FeatureObject *dominator = nullptr;

    for (auto candidate : copied)
    {
      if (dominator == nullptr)
      {
        dominator = candidate;
      }
      else if (code_index(coco::producer(candidate)) < code_index(coco::producer(dominator)))
      {
        dominator = candidate;
      }
    }

    // Replace all the occurunce of dominated objects with its dominator
    copied.erase(dominator);

    for (auto dominatee : copied)
    {
      subst(dominatee, dominator);
    }
  }
}

} // namespace enco
