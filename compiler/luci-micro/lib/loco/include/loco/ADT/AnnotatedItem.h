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

#ifndef __LOCO_ADT_ANNOTATED_ITEM_H__
#define __LOCO_ADT_ANNOTATED_ITEM_H__

#include <map>
#include <memory>
#include <typeindex>

namespace loco
{

template <typename Annotation> class AnnotatedItem
{
public:
  AnnotatedItem() = default;

public:
  virtual ~AnnotatedItem() = default;

public:
  /**
   * @brief Retrieve a stored annotation of type T
   *
   * @note This method returns nullptr if annotation does not exist
   */
  template <typename T> const T *annot(void) const
  {
    // TODO Insert static_assert(T derives Annotation);

    auto it = _attrs.find(typeid(T));

    if (it == _attrs.end())
    {
      return nullptr;
    }

    // TODO Insert null check
    return dynamic_cast<T *>(it->second.get());
  }

  /**
   * @brief Attach or remove a new annotation of type T
   *
   * @note annot<T>(nullptr) removes an attached annotation if it exists
   */
  template <typename T> void annot(std::unique_ptr<T> &&p)
  {
    // TODO: Insert static_assert(T derives Annotation);

    if (p == nullptr)
    {
      _attrs.erase(typeid(T));
    }
    else
    {
      // TODO: assert(_attribs.find(typeid(T)) == _attribs.end());
      _attrs[typeid(T)] = std::move(p);
    }
  }

private:
  std::map<std::type_index, std::unique_ptr<Annotation>> _attrs;
};

} // namespace loco

#endif // __LOCO_ADT_ANNOTATED_ITEM_H__
