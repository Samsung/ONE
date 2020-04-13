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

#include "coco/ADT/DLinkedList.h"

#include <set>

#include <gtest/gtest.h>

namespace
{

class Parent;
class Child;

using ChildList = coco::DLinkedList<Child, Parent>::Head;

class Parent
{
public:
  friend void coco::DLinkedList<Child, Parent>::joined(Parent *, Child *);
  friend void coco::DLinkedList<Child, Parent>::leaving(Parent *, Child *);

public:
  Parent() : _list{this}, _count{0}
  {
    // DO NOTHING
  }

public:
  ChildList *children(void) { return &_list; }
  uint32_t count(void) const { return _count; }

private:
  ChildList _list;
  uint32_t _count;
};

class Child final : public coco::DLinkedList<Child, Parent>::Node
{
public:
  ~Child()
  {
    if (parent())
    {
      detach();
    }
  }
};

} // namespace

namespace coco
{

template <> void DLinkedList<Child, Parent>::joined(Parent *p, Child *) { p->_count += 1; }
template <> void DLinkedList<Child, Parent>::leaving(Parent *p, Child *) { p->_count -= 1; }

template <> ChildList *DLinkedList<Child, Parent>::head(Parent *p) { return p->children(); }

} // namespace coco

namespace
{
class DLinkedListTest : public ::testing::Test
{
public:
  virtual ~DLinkedListTest()
  {
    // NOTE Child SHOULD BE freed before parent
    for (auto child : _children)
    {
      delete child;
    }

    for (auto parent : _parents)
    {
      delete parent;
    }
  }

protected:
  template <typename T> T *create(void);

  void destroy(Child *);

private:
  std::set<::Parent *> _parents;
  std::set<::Child *> _children;
};

template <>::Parent *DLinkedListTest::create(void)
{
  auto parent = new ::Parent;
  _parents.insert(parent);
  return parent;
}

template <>::Child *DLinkedListTest::create(void)
{
  auto child = new ::Child;
  _children.insert(child);
  return child;
}

void DLinkedListTest::destroy(Child *child)
{
  _children.erase(child);
  delete child;
}

} // namespace

TEST_F(DLinkedListTest, append)
{
  auto parent = create<::Parent>();
  auto child = create<::Child>();

  parent->children()->append(child);

  ASSERT_EQ(child->parent(), parent);
  ASSERT_EQ(child->prev(), nullptr);
  ASSERT_EQ(child->next(), nullptr);

  ASSERT_EQ(parent->children()->head(), child);
  ASSERT_EQ(parent->children()->tail(), child);
  ASSERT_EQ(parent->count(), 1);
}

TEST_F(DLinkedListTest, insert_two_elements)
{
  auto parent = create<::Parent>();

  ASSERT_EQ(parent->children()->head(), nullptr);
  ASSERT_EQ(parent->children()->tail(), nullptr);

  auto child_1 = create<::Child>();

  ASSERT_EQ(child_1->parent(), nullptr);
  ASSERT_EQ(child_1->prev(), nullptr);
  ASSERT_EQ(child_1->next(), nullptr);

  parent->children()->append(child_1);

  ASSERT_EQ(child_1->parent(), parent);
  ASSERT_EQ(child_1->prev(), nullptr);
  ASSERT_EQ(child_1->next(), nullptr);

  ASSERT_EQ(parent->children()->head(), child_1);
  ASSERT_EQ(parent->children()->tail(), child_1);

  auto child_2 = create<::Child>();

  ASSERT_EQ(child_2->parent(), nullptr);
  ASSERT_EQ(child_2->prev(), nullptr);
  ASSERT_EQ(child_2->next(), nullptr);

  child_2->insertAfter(child_1);

  ASSERT_EQ(child_2->parent(), parent);
  ASSERT_EQ(child_2->prev(), child_1);
  ASSERT_EQ(child_2->next(), nullptr);

  ASSERT_EQ(child_1->parent(), parent);
  ASSERT_EQ(child_1->prev(), nullptr);
  ASSERT_EQ(child_1->next(), child_2);

  ASSERT_EQ(parent->children()->head(), child_1);
  ASSERT_EQ(parent->children()->tail(), child_2);
}

TEST_F(DLinkedListTest, insertBefore)
{
  auto parent = create<::Parent>();

  auto child_1 = create<::Child>();
  auto child_2 = create<::Child>();

  parent->children()->append(child_1);
  child_2->insertBefore(child_1);

  ASSERT_EQ(child_2->parent(), parent);
  ASSERT_EQ(child_2->prev(), nullptr);
  ASSERT_EQ(child_2->next(), child_1);

  ASSERT_EQ(child_1->parent(), parent);
  ASSERT_EQ(child_1->prev(), child_2);
  ASSERT_EQ(child_1->next(), nullptr);

  ASSERT_EQ(parent->children()->head(), child_2);
  ASSERT_EQ(parent->children()->tail(), child_1);
}

TEST_F(DLinkedListTest, prepend_after_append)
{
  auto parent = create<::Parent>();

  auto child_1 = create<::Child>();
  auto child_2 = create<::Child>();

  parent->children()->append(child_1);
  parent->children()->prepend(child_2);

  ASSERT_EQ(child_2->next(), child_1);

  ASSERT_EQ(child_1->parent(), parent);
  ASSERT_EQ(child_1->prev(), child_2);
  ASSERT_EQ(child_1->next(), nullptr);

  ASSERT_EQ(parent->children()->head(), child_2);
  ASSERT_EQ(parent->children()->tail(), child_1);
}

TEST_F(DLinkedListTest, detach)
{
  auto parent = create<::Parent>();

  auto child_1 = create<::Child>();
  auto child_2 = create<::Child>();

  parent->children()->append(child_1);
  parent->children()->append(child_2);

  child_1->detach();

  ASSERT_EQ(child_1->parent(), nullptr);
  ASSERT_EQ(child_1->prev(), nullptr);
  ASSERT_EQ(child_1->next(), nullptr);

  ASSERT_EQ(child_2->parent(), parent);
  ASSERT_EQ(child_2->prev(), nullptr);

  ASSERT_EQ(parent->children()->head(), child_2);
  ASSERT_EQ(parent->children()->tail(), child_2);

  child_2->detach();

  ASSERT_EQ(child_2->parent(), nullptr);
  ASSERT_EQ(child_2->prev(), nullptr);
  ASSERT_EQ(child_2->next(), nullptr);

  ASSERT_TRUE(parent->children()->empty());
  ASSERT_EQ(parent->children()->head(), nullptr);
  ASSERT_EQ(parent->children()->tail(), nullptr);
}

TEST_F(DLinkedListTest, node_destructor)
{
  auto parent = create<::Parent>();

  auto child_1 = create<::Child>();
  auto child_2 = create<::Child>();

  parent->children()->append(child_1);
  parent->children()->append(child_2);

  destroy(child_2);

  ASSERT_EQ(parent->children()->head(), child_1);
  ASSERT_EQ(parent->children()->tail(), child_1);
  ASSERT_EQ(child_1->next(), nullptr);
  ASSERT_EQ(child_1->prev(), nullptr);

  destroy(child_1);

  ASSERT_EQ(parent->children()->head(), nullptr);
  ASSERT_EQ(parent->children()->tail(), nullptr);
}
