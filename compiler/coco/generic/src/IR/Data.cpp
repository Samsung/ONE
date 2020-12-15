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

#include "coco/IR/Data.h"

#include <nncc/core/ADT/kernel/NCHWLayout.h>
#include <nncc/core/ADT/kernel/Overlay.h>

#include <stdex/Memory.h>

#include <map>

using namespace nncc::core::ADT;

using stdex::make_unique;

namespace
{
class BlobContext
{
public:
  void allocate(const coco::Bag *b, uint32_t elemsize)
  {
    auto buffer = make_unique<std::vector<uint8_t>>();
    buffer->resize(b->size() * elemsize);

    _data[b] = std::move(buffer);
  }

  void release(const coco::Bag *b) { _data.erase(b); }

public:
  uint8_t *at(const coco::Bag *b)
  {
    auto it = _data.find(b);

    if (it != _data.end())
    {
      return it->second->data();
    }

    return nullptr;
  }

public:
  uint32_t size(const coco::Bag *b) const
  {
    auto it = _data.find(b);

    if (it != _data.end())
    {
      return it->second->size();
    }

    return 0;
  }

private:
  std::map<const coco::Bag *, std::unique_ptr<std::vector<uint8_t>>> _data;
};
} // namespace

namespace
{

template <typename T> class KernelOverlay : public kernel::Reader<T>, public kernel::Accessor<T>
{
public:
  KernelOverlay(T *base, const coco::KernelObject *object) : _base{base}, _object{object}
  {
    // DO NOTHING
  }

public:
  T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    assert(_object->layout() != nullptr);
    auto offset = _object->layout()->at(nth, ch, row, col);
    return *(_base + offset.value());
  }

public:
  T &at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) override
  {
    assert(_object->layout() != nullptr);
    auto offset = _object->layout()->at(nth, ch, row, col);
    return *(_base + offset.value());
  }

private:
  T *_base;
  const coco::KernelObject *_object;
};

} // namespace

namespace
{
template <typename T> class PlainWeightContextImpl final : public coco::PlainWeightContext<T>
{
public:
  PlainWeightContextImpl(BlobContext *blob) : _blob{blob}
  {
    // DO NOTHING
  }

public:
  PlainWeightContextImpl(const PlainWeightContextImpl &) = delete;
  PlainWeightContextImpl(PlainWeightContextImpl &&) = delete;

public:
  coco::Span<T> allocate(const coco::Bag *bag) override
  {
    assert(bag != nullptr);
    _blob->allocate(bag, sizeof(T));
    return weight(bag);
  }

  coco::Span<T> weight(const coco::Bag *b) override
  {
    // TODO Check type later
    if (auto data = _blob->at(b))
    {
      uint32_t byte_size = _blob->size(b);
      assert(byte_size % sizeof(T) == 0);
      uint32_t elem_size = static_cast<uint32_t>(byte_size / sizeof(T));

      return coco::Span<T>{reinterpret_cast<T *>(data), elem_size};
    }

    return coco::Span<T>{nullptr, 0};
  }

public:
  std::unique_ptr<kernel::Accessor<T>> access(const coco::KernelObject *o) override
  {
    auto b = o->bag();
    assert(b != nullptr);

    if (auto base = reinterpret_cast<T *>(_blob->at(b)))
    {
      return make_unique<KernelOverlay<T>>(base, o);
    }

    return nullptr;
  }

public:
  std::unique_ptr<kernel::Reader<T>> read(const coco::KernelObject *o) const override
  {
    auto b = o->bag();
    assert(b != nullptr);

    if (auto base = reinterpret_cast<T *>(_blob->at(b)))
    {
      return make_unique<KernelOverlay<T>>(base, o);
    }

    return nullptr;
  }

private:
  BlobContext *const _blob;
};
} // namespace

namespace
{
struct DataImpl final : public coco::Data
{
  std::unique_ptr<BlobContext> _blob;
  std::unique_ptr<PlainWeightContextImpl<float>> _fp32;

  bool allocated(const coco::Bag *b) const override { return _blob->at(b) != nullptr; }

  void release(const coco::Bag *b) override
  {
    assert(allocated(b));
    _blob->release(b);
  }

  coco::PlainWeightContext<float> *f32(void) override { return _fp32.get(); }
  const coco::PlainWeightContext<float> *f32(void) const override { return _fp32.get(); }
};
} // namespace

namespace coco
{

std::unique_ptr<Data> Data::create(void)
{
  auto blob = make_unique<BlobContext>();
  auto fp32 = make_unique<PlainWeightContextImpl<float>>(blob.get());

  auto data = make_unique<DataImpl>();

  data->_blob = std::move(blob);
  data->_fp32 = std::move(fp32);

  // GCC 4.9 tries to copy data (while GCC 6.X doesn't)
  return std::move(data);
}

} // namespace coco
