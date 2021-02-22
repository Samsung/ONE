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

#include "loco/ADT/AnnotatedItem.h"

#include <gtest/gtest.h>

#include <memory>

namespace
{

struct Annotation
{
  virtual ~Annotation() = default;
};

template <int N> struct DerivedAnnotation final : public Annotation
{
  static std::unique_ptr<DerivedAnnotation<N>> make(void)
  {
    return std::make_unique<DerivedAnnotation<N>>();
  }
};

} // namespace

TEST(AnnotatedItemTest, annotation)
{
  loco::AnnotatedItem<::Annotation> item;

  ASSERT_EQ(nullptr, item.annot<DerivedAnnotation<0>>());

  item.annot(DerivedAnnotation<0>::make());

  ASSERT_NE(item.annot<DerivedAnnotation<0>>(), nullptr);
  ASSERT_EQ(nullptr, item.annot<DerivedAnnotation<1>>());

  item.annot<DerivedAnnotation<0>>(nullptr);
  ASSERT_EQ(nullptr, item.annot<DerivedAnnotation<0>>());

  // Below check guarantees that "annot<T>(nullptr)" is allowed even when there is no annotation.
  // This guarantee allows us to simplify code for some cases.
  //
  // Let us consider the following example:
  //
  // void f(loco::AnnotatedItem<T> *item)
  // {
  //   /* DO SOMETHING */
  //   if (cond) { item->annot<T>(nullptr);
  // }
  //
  // void g(loco::AnnotatedItem<T> *item)
  // {
  //   f(item);
  //   item->annot<T>(nullptr);
  // }
  //
  // The implementation of "g" gets complicated if annot<T>(nullptr) is not allowed if there is
  // no annotation.
  //
  item.annot<DerivedAnnotation<0>>(nullptr);
}
