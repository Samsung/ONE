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

#ifndef __COCO_ADT_DLINKED_LIST_H__
#define __COCO_ADT_DLINKED_LIST_H__

#include <cassert>
#include <type_traits>

namespace coco
{

// **CAUTION** Child SHOULD inherit DLinkedList<Child, Parent>::Node
template <typename Child, typename Parent> struct DLinkedList
{
  /// @brief A hook for Child-Join event
  static void joined(Parent *, Child *);
  /// @brief A hook for Child-Leave event
  static void leaving(Parent *, Child *);

  class Head
  {
  public:
    Head(Parent *parent) : _parent{parent}
    {
      _head = nullptr;
      _tail = nullptr;
    }

  public:
    Head(const Head &) = delete;
    Head(Head &&) = delete;

  public:
    Child *head(void) const { return _head; }
    Child *tail(void) const { return _tail; }

  public:
    bool empty(void) const
    {
      if (_head == nullptr)
      {
        assert(_head == _tail);
        return true;
      }

      assert(_head != nullptr);
      assert(_tail != nullptr);
      return false;
    }

  public:
    void enlist(Child *child)
    {
      assert((child->prev() == nullptr) || (child->prev()->parent() == _parent));
      assert((child->next() == nullptr) || (child->next()->parent() == _parent));

      if (empty())
      {
        _head = child;
        _tail = child;
      }
      else
      {
        if (child->next() == _head)
        {
          // _child is a new head
          assert(child->prev() == nullptr);
          _head = child;
        }

        if (child->prev() == _tail)
        {
          // _child is a new tail
          assert(child->next() == nullptr);
          _tail = child;
        }
      }

      // Update parent-child relation
      child->parent(_parent);

      // Notify Child-Joining event
      joined(_parent, child);
    }

  public:
    void delist(Child *child)
    {
      assert(child->parent() == _parent);
      assert(!empty());

      // Notify Child-Leaving event
      leaving(_parent, child);

      if (child == _head)
      {
        _head = child->next();
      }

      if (child == _tail)
      {
        _tail = child->prev();
      }

      // Update parent-child relation
      child->parent(nullptr);
    }

  public:
    void prepend(Child *child)
    {
      if (empty())
      {
        enlist(child);
      }
      else
      {
        child->insertBefore(_head);
      }
    }

  public:
    void append(Child *child)
    {
      if (empty())
      {
        enlist(child);
      }
      else
      {
        child->insertAfter(_tail);
      }
    }

  private:
    Parent *const _parent;

  private:
    Child *_head;
    Child *_tail;
  };

  // NOTE Client SHOULD implement this static method
  static Head *head(Parent *);

  class Node
  {
  public:
    friend class Head;

  public:
    Node()
    {
      static_assert(std::is_base_of<Node, Child>::value,
                    "Type `Child` must be subclass of `Node`.");

      _prev = nullptr;
      _next = nullptr;
    }

  public:
    virtual ~Node()
    {
      // Each Child should unlink itself on destruction
      //
      // NOTE detach invokes "leaving" hook which may access the internal of each Child,
      //      so it is not safe to invoke detach here
      assert(parent() == nullptr);
    }

  public:
    Parent *parent(void) const { return _parent; }

  private:
    Child *curr(void) { return reinterpret_cast<Child *>(this); }
    const Child *curr(void) const { return reinterpret_cast<const Child *>(this); }

  public:
    Child *prev(void) const { return _prev; }
    Child *next(void) const { return _next; }

  public:
    void insertBefore(Node *next)
    {
      assert(next != nullptr);
      assert(next->parent() != nullptr);
      assert(head(next->parent()) != nullptr);

      assert(_prev == nullptr);
      assert(_next == nullptr);

      // Update the link of the current node
      _prev = next->prev();
      _next = next->curr();

      if (auto prev = next->prev())
      {
        prev->_next = curr();
      }
      next->_prev = curr();

      // Update parent-child relation
      assert(parent() == nullptr);
      head(next->parent())->enlist(curr());
      assert(parent() == next->parent());
    }

  public:
    void insertAfter(Node *prev)
    {
      assert(prev != nullptr);
      assert(prev->parent() != nullptr);
      assert(head(prev->parent()) != nullptr);

      assert(_prev == nullptr);
      assert(_next == nullptr);

      // Update the link of the current node
      _prev = prev->curr();
      _next = prev->next();

      // Update the link of the sibling nodes
      if (auto next = prev->next())
      {
        next->_prev = curr();
      }
      prev->_next = curr();

      // Update parent-child relation
      assert(parent() == nullptr);
      head(prev->parent())->enlist(curr());
      assert(parent() == prev->parent());
    };

  public:
    void detach(void)
    {
      // Update parent-child relation
      assert(parent() != nullptr);
      assert(head(parent()) != nullptr);
      head(parent())->delist(curr());
      assert(parent() == nullptr);

      // Update the link of sibling nodes
      if (prev())
      {
        prev()->_next = next();
      }

      if (next())
      {
        next()->_prev = prev();
      }

      // Update the link of the current node
      _prev = nullptr;
      _next = nullptr;
    }

  private:
    // WARN Do NOT invoke this method outside Head::enlist
    void parent(Parent *p) { _parent = p; }

  private:
    // WARN Do NOT modify this field inside Node.
    Parent *_parent = nullptr;
    Child *_prev;
    Child *_next;
  };
};

} // namespace coco

#endif // __COCO_ADT_DLINKED_LIST_H__
