/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_TRAIN_LAYER_SCOPE_TENSOR_INDEX_H__
#define __ONERT_BACKEND_TRAIN_LAYER_SCOPE_TENSOR_INDEX_H__

#include <ir/Index.h>

#include <cassert>

namespace onert::backend::train
{

class LayerScopeTensorIndex
{
public:
  LayerScopeTensorIndex(const ir::OperationIndex &op_index, uint32_t sub_index)
    : _op_index{op_index}, _sub_index{sub_index}
  {
    assert(op_index.valid());
  }

public:
  const ir::OperationIndex &op_index() const { return _op_index; }
  uint32_t sub_index() const { return _sub_index; }

  bool operator==(const LayerScopeTensorIndex &other) const
  {
    return _op_index == other.op_index() && _sub_index == other.sub_index();
  }
  bool operator!=(const LayerScopeTensorIndex &other) const { return !(*this == other); }

private:
  ir::OperationIndex _op_index;
  uint32_t _sub_index;
};

inline std::ostream &operator<<(std::ostream &o, const LayerScopeTensorIndex &i)
{
  o << i.op_index() << "-" << i.sub_index();
  return o;
}

} // namespace onert::backend::train

namespace std
{

template <> struct hash<onert::backend::train::LayerScopeTensorIndex>
{
  size_t operator()(const onert::backend::train::LayerScopeTensorIndex &index) const noexcept
  {
    const auto op_index = index.op_index();
    const auto sub_index = index.sub_index();

    assert(sizeof(op_index) <= sizeof(uint32_t));
    static_assert(sizeof(size_t) >= sizeof(uint32_t),
                  "LayerScopeTensorIndex's hash creation error, size_t size is less than uint32_t");

    return (static_cast<size_t>(op_index.value())) << 16 | static_cast<size_t>(sub_index);
  }
};

} // namespace std

#endif // __ONERT_BACKEND_TRAIN_LAYER_SCOPE_TENSOR_INDEX_H__
