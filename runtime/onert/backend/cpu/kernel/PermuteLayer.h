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

#ifndef __ONERT_BACKEND_CPU_KERNEL_PERMUTE_LAYER_H__
#define __ONERT_BACKEND_CPU_KERNEL_PERMUTE_LAYER_H__

#include "OperationUtils.h"

#include <exec/IFunction.h>
#include <ir/Coordinates.h>
#include <ir/operation/Permute.h>
#include <misc/feature/IndexIterator.h>
#include <util/feature/nchw/View.h>
#include <util/feature/nhwc/Reader.h>
#include <util/feature/nhwc/View.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class PermuteLayer : public ::onert::exec::IFunction
{
public:
  PermuteLayer() = default;

public:
  void configure(std::shared_ptr<backend::ITensor> input, std::shared_ptr<backend::ITensor> output,
                 const ir::Shape &output_shape, ir::operation::Permute::Type type,
                 ir::DataType dataType);
  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  template <class T> void runTempl()
  {
    auto rank = _output_shape.rank();
    auto fn = [&](ITensor &in_tensor) {
      _output->access([&](ITensor &out_tensor) {
        auto input_buffer = in_tensor.buffer();
        auto input_size = in_tensor.total_size();
        auto output_buffer = out_tensor.buffer();
        if (_type == ir::operation::Permute::Type::COPY)
        {
          assert(in_tensor.layout() == out_tensor.layout());
          if (!in_tensor.has_padding() && !out_tensor.has_padding())
          {
            assert(input_size == out_tensor.total_size());
            memcpy(output_buffer, input_buffer, input_size);
            return;
          }
        }
        switch (rank)
        {
          case 0:
          case 1:
          {
            const int32_t copy_len = _output_shape.dim(0);

            memcpy(output_buffer, input_buffer, copy_len);
            break;
          }
          case 2:
          {
            const int32_t copy_len = _output_shape.dim(1);

            for (auto i = 0; i < _output_shape.dim(0); ++i)
            {
              ir::Coordinates coords{i, 0};
              memcpy(output_buffer + out_tensor.calcOffset(coords),
                     input_buffer + in_tensor.calcOffset(coords), copy_len * sizeof(T));
            }
            break;
          }
          case 3:
          {
            const int32_t copy_len = _output_shape.dim(2);

            for (auto i = 0; i < _output_shape.dim(0); ++i)
            {
              for (auto j = 0; j < _output_shape.dim(1); ++j)
              {
                ir::Coordinates coords{i, j, 0};
                memcpy(output_buffer + out_tensor.calcOffset(coords),
                       input_buffer + in_tensor.calcOffset(coords), copy_len * sizeof(T));
              }
            }
            break;
          }
          case 4:
          {
            // TODO Unify permute type and remove switch case
            switch (_type)
            {
              case ir::operation::Permute::Type::NHWC_TO_NCHW:
              {
                for (auto n = 0; n < _output_shape.dim(0); ++n)
                {
                  for (auto c = 0; c < _output_shape.dim(1); ++c)
                  {
                    for (auto h = 0; h < _output_shape.dim(2); ++h)
                    {
                      for (auto w = 0; w < _output_shape.dim(3); ++w)
                      {
                        const ir::Coordinates in_coords{n, h, w, c};
                        const auto out_coords =
                            convertCoordinates(in_coords, in_tensor.layout(), out_tensor.layout());
                        const auto value =
                            *reinterpret_cast<T *>(input_buffer + in_tensor.calcOffset(in_coords));
                        *reinterpret_cast<T *>(output_buffer + out_tensor.calcOffset(out_coords)) =
                            value;
                      }
                    }
                  }
                }
                break;
              }
              case ir::operation::Permute::Type::NCHW_TO_NHWC:
              {
                for (auto n = 0; n < _output_shape.dim(0); ++n)
                {
                  for (auto h = 0; h < _output_shape.dim(1); ++h)
                  {
                    for (auto w = 0; w < _output_shape.dim(2); ++w)
                    {
                      for (auto c = 0; c < _output_shape.dim(3); ++c)
                      {
                        const ir::Coordinates in_coords{n, c, h, w};
                        const auto out_coords =
                            convertCoordinates(in_coords, in_tensor.layout(), out_tensor.layout());
                        const auto value =
                            *reinterpret_cast<T *>(input_buffer + in_tensor.calcOffset(in_coords));
                        *reinterpret_cast<T *>(output_buffer + out_tensor.calcOffset(out_coords)) =
                            value;
                      }
                    }
                  }
                }
                break;
              }
              case ir::operation::Permute::Type::COPY:
              {
                const int32_t copy_len = _output_shape.dim(3);

                for (auto i = 0; i < _output_shape.dim(0); ++i)
                {
                  for (auto j = 0; j < _output_shape.dim(1); ++j)
                  {
                    for (auto k = 0; k < _output_shape.dim(2); ++k)
                    {
                      ir::Coordinates coords{i, j, k, 0};
                      memcpy(output_buffer + out_tensor.calcOffset(coords),
                             input_buffer + in_tensor.calcOffset(coords), copy_len * sizeof(T));
                    }
                  }
                }
                break;
              }
              default:
                throw std::runtime_error("NYI");
                break;
            }
            break;
          }
          default:
            throw std::runtime_error("NYI");
            break;
        }
      });
    };
    _input->access(fn);
  }

private:
  std::shared_ptr<backend::ITensor> _input{nullptr};
  std::shared_ptr<backend::ITensor> _output{nullptr};
  ir::Shape _output_shape{};
  ir::operation::Permute::Type _type{ir::operation::Permute::Type::COPY};
  ir::DataType _dataType{ir::DataType::FLOAT32};
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_PERMUTE_LAYER_H__
