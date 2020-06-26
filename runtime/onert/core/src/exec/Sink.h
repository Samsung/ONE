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

#ifndef __ONERT_EXEC_SINK_H__
#define __ONERT_EXEC_SINK_H__

#include "feature/nchw/Reader.h"
#include "feature/nchw/View.h"
#include "feature/nhwc/Reader.h"
#include "feature/nhwc/View.h"

#include <cassert>
#include <memory>
#include "util/Utils.h"
#include <misc/feature/IndexIterator.h>

namespace onert
{
namespace exec
{
struct ISink
{
  virtual ~ISink() = default;

  virtual void pull(::onert::backend::ITensor &tensor) const = 0;
};

// Create second lever inheritance: the first lever is used as a reference type in use-case places
template <typename T> class ITemplSink : public ISink
{
public:
  ITemplSink(void *output_buffer, const size_t &output_size, const ir::Shape &shape,
             const bool copy, ir::Layout io_layout)
      : _output_buffer{reinterpret_cast<T *>(output_buffer)}, _output_size{output_size},
        _shape{shape}, _copy{copy}, _io_layout{io_layout}
  {
  }

protected:
  void pullUnif(onert::backend::ITensor &tensor) const
  {
    assert(((_io_layout == ir::Layout::NHWC && tensor.layout() == ir::Layout::NCHW) ||
            (_io_layout == ir::Layout::NCHW && tensor.layout() == ir::Layout::NHWC)) ||
           _copy);
    auto input_buffer = tensor.buffer();
    auto rank = _shape.rank();

    if (!tensor.has_padding() && rank < 4 + _copy)
    {
      memcpy(_output_buffer, input_buffer, _output_size);
      return;
    }

    switch (rank)
    {
      case 0:
      case 1:
      {
        memcpy(_output_buffer, input_buffer, _output_size);
        break;
      }
      case 2:
      {
        const int32_t copy_len = _shape.dim(1);

        for (auto i = 0; i < _shape.dim(0); ++i)
        {
          ir::Coordinates coords{i, 0};
          memcpy(_output_buffer + i * copy_len, input_buffer + tensor.calcOffset(coords),
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
            memcpy(_output_buffer + i * dim1 * dim2 + j * dim2,
                   input_buffer + tensor.calcOffset(coords), dim2 * sizeof(T));
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
                memcpy(_output_buffer + i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3,
                       input_buffer + tensor.calcOffset(coords), dim3 * sizeof(T));
              }
            }
          }
        }
        else
        {
          const auto shape = _shape.asFeature(_io_layout);

          if (_io_layout == ir::Layout::NHWC)
          {
            const exec::feature::nchw::Reader<T> from(&tensor);
            exec::feature::nhwc::View<T> into(shape, _output_buffer, _output_size);
            ::nnfw::misc::feature::iterate(shape)
                << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
                     const auto value = from.at(batch, ch, row, col);
                     into.at(batch, row, col, ch) = value;
                   };
          }
          else if (_io_layout == ir::Layout::NCHW)
          {
            const exec::feature::nhwc::Reader<T> from(&tensor);
            exec::feature::nchw::View<T> into(shape, _output_buffer, _output_size);
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
  T *_output_buffer;
  const size_t _output_size;
  const ir::Shape _shape;
  const bool _copy;
  const ir::Layout _io_layout;
};

template <typename T> class PermutateSink final : public ITemplSink<T>
{
public:
  PermutateSink(void *output_buffer, const size_t &output_size, const ir::Shape &shape,
                ir::Layout io_layout)
      : ITemplSink<T>(output_buffer, output_size, shape, false, io_layout)
  {
  }

public:
  void pull(onert::backend::ITensor &tensor) const override { ITemplSink<T>::pullUnif(tensor); }
};

// Only supports NHWC format front-end(NNAPI) now
template <typename T> class CopySink final : public ITemplSink<T>
{
public:
  CopySink(void *output_buffer, const size_t &output_size, const ir::Shape &shape,
           ir::Layout io_layout = ir::Layout::UNKNOWN)
      : ITemplSink<T>(output_buffer, output_size, shape, true, io_layout)
  {
  }

public:
  void pull(onert::backend::ITensor &tensor) const override { ITemplSink<T>::pullUnif(tensor); }
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_SINK_H__
