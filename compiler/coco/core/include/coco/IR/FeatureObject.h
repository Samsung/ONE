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

#ifndef __COCO_IR_FEATURE_OBJECT_H__
#define __COCO_IR_FEATURE_OBJECT_H__

#include "coco/IR/Object.h"
#include "coco/IR/FeatureShape.h"
#include "coco/IR/FeatureLayout.h"
#include "coco/IR/ElemID.h"

#include <nncc/core/ADT/feature/Layout.h>

#include <vector>

namespace coco
{

/**
 * @brief FeatureMap values (used in CNN)
 */
class FeatureObject final : public Object
{
public:
  FeatureObject() = default;

public:
  ~FeatureObject();

public:
  Object::Kind kind(void) const override { return Object::Kind::Feature; }

public:
  FeatureObject *asFeature(void) override { return this; }
  const FeatureObject *asFeature(void) const override { return this; }

public:
  const FeatureShape &shape(void) const;

public:
  const FeatureLayout *layout(void) const { return _layout.get(); }
  void layout(std::unique_ptr<FeatureLayout> &&l) { _layout = std::move(l); }

private:
  std::unique_ptr<FeatureLayout> _layout;
};

} // namespace coco

#endif // __COCO_IR_FEATURE_OBJECT_H__
