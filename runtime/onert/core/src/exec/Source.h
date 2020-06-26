/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_SOURCE_H__
#define __ONERT_EXEC_SOURCE_H__

#include "feature/nchw/Reader.h"
#include "feature/nchw/View.h"
#include "feature/nhwc/Reader.h"
#include "feature/nhwc/View.h"

#include <cassert>
#include <memory>
#include "util/Utils.h"
#include <misc/feature/IndexIterator.h>
#include <ir/Layout.h>
#include "ir/Shape.h"

namespace onert
{
namespace exec
{

struct ISource
{
  virtual ~ISource() = default;

  virtual void push(::onert::backend::ITensor &tensor) const = 0;
};

// Create second lever inheritance: the first lever is used as a reference type in use-case places
template <typename T> class ITemplSource : public ISource
{
public:
  ITemplSource(const void *input_buffer, const size_t &input_size, const ir::Shape &shape,
               const bool copy, ir::Layout io_layout)
      : _input_buffer{reinterpret_cast<const T *>(input_buffer)}, _input_size{input_size},
        _shape{shape}, _copy(copy), _io_layout{io_layout}
  {
  }

  virtual void push(::onert::backend::ITensor &tensor) const = 0;

protected:
  void pushUnif(onert::backend::ITensor &tensor) const
  {
    assert(((_io_layout == ir::Layout::NHWC && tensor.layout() == ir::Layout::NCHW) ||
            (_io_layout == ir::Layout::NCHW && tensor.layout() == ir::Layout::NHWC)) ||
           _copy);
    auto output_buffer = tensor.buffer();
    auto rank = _shape.rank();

    if (!tensor.has_padding() && rank < 4 + _copy)
    {
      memcpy(output_buffer, _input_buffer, _input_size);
      return;
    }

    switch (rank)
    {
      case 0:
      case 1:
      {
        memcpy(output_buffer, _input_buffer, _input_size);
        break;
      }
      case 2:
      {
        const int32_t copy_len = _shape.dim(1);

        for (auto i = 0; i < _shape.dim(0); ++i)
        {
          ir::Coordinates coords{i, 0};
          memcpy(output_buffer + tensor.calcOffset(coords), _input_buffer + i * copy_len,
                 copy_len * sizeof(T));
        }
        break;
      }
      case 3:
      {
        const int32_t dim1 = _shape.dim(1);
        const int32_t dim2 = _shape.dim(2);

        for (auto i = 0; i < _shape.dim(0); ++i)
        {
          for (auto j = 0; j < _shape.dim(1); ++j)
          {
            ir::Coordinates coords{i, j, 0};
            memcpy(output_buffer + tensor.calcOffset(coords),
                   _input_buffer + i * dim1 * dim2 + j * dim2, dim2 * sizeof(T));
          }
        }
        break;
      }
      case 4:
      {
        if (_copy)
        {
          const int32_t dim1 = _shape.dim(1);
          const int32_t dim2 = _shape.dim(2);
          const int32_t dim3 = _shape.dim(3);
          for (auto i = 0; i < _shape.dim(0); ++i)
          {
            for (auto j = 0; j < _shape.dim(1); ++j)
            {
              for (auto k = 0; k < _shape.dim(2); ++k)
              {
                ir::Coordinates coords{i, j, k, 0};
                memcpy(output_buffer + tensor.calcOffset(coords),
                       _input_buffer + i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3,
                       dim3 * sizeof(T));
              }
            }
          }
        }
        else
        {
          const auto shape = _shape.asFeature(_io_layout);

          if (_io_layout == ir::Layout::NCHW)
          {
            const exec::feature::nchw::Reader<T> from(shape, _input_buffer, _input_size);
            exec::feature::nhwc::View<T> into(&tensor);
            ::nnfw::misc::feature::iterate(shape)
                << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
                     const auto value = from.at(batch, ch, row, col);
                     into.at(batch, row, col, ch) = value;
                   };
          }
          else if (_io_layout == ir::Layout::NHWC)
          {
            const exec::feature::nhwc::Reader<T> from(shape, _input_buffer, _input_size);
            exec::feature::nchw::View<T> into(&tensor);
            ::nnfw::misc::feature::iterate(shape)
                << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
                     const auto value = from.at(batch, row, col, ch);
                     into.at(batch, ch, row, col) = value;
                   };
          }
          else
          {
            throw std::runtime_error("Wrong Layout");
          }
        }

        break;
      }
      default:
        throw std::runtime_error("NYI: rank > 4");
        break;
    }
  }

private:
  const T *_input_buffer;
  const size_t _input_size;
  const ir::Shape _shape;
  const bool _copy;
  const ir::Layout _io_layout;
};

template <typename T> class PermutateSource final : public ITemplSource<T>
{
public:
  PermutateSource(const void *input_buffer, const size_t &input_size, const ir::Shape &shape,
                  ir::Layout io_layout)
      : ITemplSource<T>(input_buffer, input_size, shape, false, io_layout)
  {
  }

public:
  void push(onert::backend::ITensor &tensor) const override
  {
    // do NHWC_TO_NCHW or NCHW_TO_NHWC permutation
    ITemplSource<T>::pushUnif(tensor);
  }
};

template <typename T> class CopySource final : public ITemplSource<T>
{
public:
  CopySource(const void *input_buffer, const size_t &input_size, const ir::Shape &shape,
             ir::Layout io_layout = ir::Layout::UNKNOWN)
      : ITemplSource<T>(input_buffer, input_size, shape, true, io_layout)
  {
  }

public:
  void push(onert::backend::ITensor &tensor) const override { ITemplSource<T>::pushUnif(tensor); }
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_SOURCE_H__
