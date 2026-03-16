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

#include "FeatureUnification.h"
#include "IRUtils.h"

#include <memory>
#include <set>
#include <vector>

#include <cassert>

using std::make_unique;

namespace
{

bool is_static_layout(const coco::FeatureLayout::ID *id)
{
  if (id == coco::FeatureLayouts::BHWC::uid())
  {
    return true;
  }

  if (id == coco::FeatureLayouts::BCHW::uid())
  {
    return true;
  }

  return false;
}

bool is_static_layout(const coco::FeatureLayout *l) { return is_static_layout(l->id()); }
bool is_static_layout(const coco::FeatureObject *f) { return is_static_layout(f->layout()); }

/**
 * @brief Return ture if a given 'feature' is the candidate of unification
 */
bool candidate(const coco::FeatureObject *f) { return is_static_layout(f); }

/**
 * @brief Return true if two features are compatible
 *
 * Two features are referred to as compatible if these feature are interchangeable.
 *
 * NOTE The current implementation of "compatible" is sound, but incomplete.
 *
 * Soundness:
 *  For all feature objects "lhs" and "rhs" that "compatible(lhs, rhs)" returns true,
 *  "lhs" and "rhs" are interchangeable.
 *
 * Completeness:
 *   For all interchangeable feature objects "lhs" and "rhs", "compatible(lhs, rhs)" returns true.
 */
bool compatible(const coco::FeatureObject *lhs, const coco::FeatureObject *rhs)
{
  assert(candidate(lhs) && candidate(rhs));

  if (lhs->layout()->id() != rhs->layout()->id())
  {
    return false;
  }

  if (lhs->layout()->batch() != rhs->layout()->batch())
  {
    return false;
  }

  if (!(lhs->layout()->shape() == rhs->layout()->shape()))
  {
    return false;
  }

  return true;
}

/**
 * @brief A FeatureGroup denotes a group of FeatureObject(s)
 *
 * Each FeatureGroup includes at most 1 DEF FeatureObject (a FeatureObject that has a producer),
 * and may include multiple USE FeatureObject(s) (a FeatureObject that has no producer).
 *
 * NOTE FeatureUnification pass internally uses this FeatureGroup to store a group of compatible
 *      FeatureObject(s)
 */
class FeatureGroup
{
public:
  explicit FeatureGroup(coco::FeatureObject *feature) { insert(feature); }

public:
  uint32_t size(void) const { return _uses.size() + (_def ? 1 : 0); }

public:
  void insert(coco::FeatureObject *feature)
  {
    if (feature->def() != nullptr)
    {
      assert(_def == nullptr);
      _def = feature;
    }
    else
    {
      _uses.insert(feature);
    }
  }

public:
  coco::FeatureObject *parent(void) const
  {
    if (_def)
    {
      return _def;
    }

    assert(_uses.size() > 0);
    return *(_uses.begin());
  }

public:
  std::set<coco::FeatureObject *> children(void) const
  {
    auto res = _uses;
    res.erase(parent());
    return res;
  }

private:
  coco::FeatureObject *_def = nullptr;
  std::set<coco::FeatureObject *> _uses;
};

} // namespace

namespace enco
{

void unify_feature(enco::Code *code)
{
  auto m = code->module();

  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    std::vector<std::unique_ptr<FeatureGroup>> groups;

    auto assign_group = [&](coco::FeatureObject *feature) {
      // Find a compatible FeatureGroup
      FeatureGroup *group = nullptr;

      for (const auto &g : groups)
      {
        FeatureGroup *candidate = g.get();

        if (!compatible(candidate->parent(), feature))
        {
          continue;
        }

        group = candidate;
        break;
      }

      if (group == nullptr)
      {
        // Insert FeatureObject into a new FeatureGroup
        groups.emplace_back(make_unique<FeatureGroup>(feature));
      }
      else
      {
        // Insert FeatureObject into the compatible FeatureGroup
        group->insert(feature);
      }
    };

    auto bag = m->entity()->bag()->at(n);

    for (auto o : coco::dependent_objects(bag))
    {
      if (auto feature = o->asFeature())
      {
        if (candidate(feature))
        {
          assign_group(feature);
        }
      }
    }

    for (const auto &g : groups)
    {
      auto group = g.get();
      for (const auto child : group->children())
      {
        subst(child, group->parent());
        assert(child->def() == nullptr);
        assert(child->uses()->size() == 0);
        m->entity()->object()->destroy(child);
      }
    }
  }
}

} // namespace enco
