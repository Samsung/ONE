/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_I_PERMUTE_FUNCTION_H__
#define __ONERT_EXEC_I_PERMUTE_FUNCTION_H__

#include "backend/ITensor.h"
#include "exec/IFunction.h"
#include "ir/Index.h"
#include "ir/Shape.h"
#include <memory>
#include <misc/feature/IndexIterator.h>
#include <typeinfo>
#include "util/feature/nchw/Reader.h"
#include "util/feature/nchw/View.h"
#include "util/feature/nhwc/Reader.h"
#include "util/feature/nhwc/View.h"
#include "util/Utils.h"
#include <vector>

namespace onert
{
namespace exec
{

class IPermuteFunction : public IFunction
{
private:
  enum class PermuteType
  {
    NHWC_TO_NCHW,
    NCHW_TO_NHWC,
    COPY
  };

public:
  virtual void run() override
  {
    assert(_src_tensors.size() > 0);
    assert(_src_tensors.size() == _dst_tensors.size());
    auto src_it = _src_tensors.begin();
    auto dst_it = _dst_tensors.begin();
    while (src_it != _src_tensors.end())
    {
      const auto src_tensor = *src_it;
      auto dst_tensor = *dst_it;
      if (src_tensor != dst_tensor)
      {
        // TODO Change to permute in parallel
        assert(underlying_type(src_tensor->data_type()) ==
               underlying_type(dst_tensor->data_type()));
        const auto rank = src_tensor->num_dimensions();
        switch (src_tensor->data_type())
        {
          case ir::DataType::FLOAT32:
            permute<float>(src_tensor, dst_tensor, rank);
            break;
          case ir::DataType::INT32:
            permute<int32_t>(src_tensor, dst_tensor, rank);
            break;
          case ir::DataType::UINT32:
            permute<uint32_t>(src_tensor, dst_tensor, rank);
            break;
          case ir::DataType::BOOL8:
          case ir::DataType::QUANT_UINT8_ASYMM:
          case ir::DataType::UINT8:
            permute<uint8_t>(src_tensor, dst_tensor, rank);
            break;
          case ir::DataType::QUANT_INT8_SYMM:
            permute<int8_t>(src_tensor, dst_tensor, rank);
            break;
          case ir::DataType::INT64:
            permute<int64_t>(src_tensor, dst_tensor, rank);
            break;
          default:
            throw std::runtime_error("IPermuteFunction: Not supported data type");
            break;
        }
      }
      src_it++;
      dst_it++;
    }
  }

  virtual void prepare() override { optimize(); }

  virtual void optimize() = 0;

private:
  template <class T>
  void permute(const std::shared_ptr<backend::ITensor> &src, std::shared_ptr<backend::ITensor> &dst,
               size_t rank)
  {
    const auto permute_type = [&]() -> PermuteType {
      if (src->layout() == ir::Layout::NHWC && dst->layout() == ir::Layout::NCHW)
      {
        return PermuteType::NHWC_TO_NCHW;
      }
      else if (src->layout() == ir::Layout::NCHW && dst->layout() == ir::Layout::NHWC)
      {
        return PermuteType::NCHW_TO_NHWC;
      }
      else
      {
        return PermuteType::COPY;
      }
    }();
    auto fn = [&](backend::ITensor &src_tensor) {
      dst->access([&](backend::ITensor &dst_tensor) {
        auto src_buffer = src_tensor.buffer();
        auto src_size = src_tensor.total_size();
        auto dst_buffer = dst_tensor.buffer();
        if (permute_type == PermuteType::COPY)
        {
          assert(src_tensor.layout() == dst_tensor.layout());
          if (!src_tensor.has_padding() && !dst_tensor.has_padding())
          {
            assert(src_size == dst_tensor.total_size());
            memcpy(dst_buffer, src_buffer, src_size);
            return;
          }
        }
        switch (rank)
        {
          case 0:
          case 1:
          {
            const int32_t copy_len = dst_tensor.dimension(0);

            memcpy(dst_buffer, src_buffer, copy_len);
            break;
          }
          case 2:
          {
            const int32_t dim_0 = dst_tensor.dimension(0);
            const int32_t copy_len = dst_tensor.dimension(1);

            for (int32_t i = 0; i < dim_0; ++i)
            {
              ir::Coordinates coords{i, 0};
              memcpy(dst_buffer + dst_tensor.calcOffset(coords),
                     src_buffer + src_tensor.calcOffset(coords), copy_len * sizeof(T));
            }
            break;
          }
          case 3:
          {
            const int32_t dim_0 = dst_tensor.dimension(0);
            const int32_t dim_1 = dst_tensor.dimension(1);
            const int32_t copy_len = dst_tensor.dimension(2);

            for (auto i = 0; i < dim_0; ++i)
            {
              for (auto j = 0; j < dim_1; ++j)
              {
                ir::Coordinates coords{i, j, 0};
                memcpy(dst_buffer + dst_tensor.calcOffset(coords),
                       src_buffer + src_tensor.calcOffset(coords), copy_len * sizeof(T));
              }
            }
            break;
          }
          case 4:
          {
            switch (permute_type)
            {
              case PermuteType::NHWC_TO_NCHW:
              {
                ir::FeatureShape shape;
                shape.N = dst_tensor.dimension(0);
                shape.C = dst_tensor.dimension(1);
                shape.H = dst_tensor.dimension(2);
                shape.W = dst_tensor.dimension(3);
                const util::feature::nhwc::Reader<T> from(&src_tensor);
                util::feature::nchw::View<T> into(&dst_tensor);
                ::nnfw::misc::feature::iterate(shape)
                    << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
                         const auto value = from.at(batch, row, col, ch);
                         into.at(batch, ch, row, col) = value;
                       };
                break;
              }
              case PermuteType::NCHW_TO_NHWC:
              {
                ir::FeatureShape shape;
                shape.N = src_tensor.dimension(0);
                shape.C = src_tensor.dimension(1);
                shape.H = src_tensor.dimension(2);
                shape.W = src_tensor.dimension(3);
                const util::feature::nchw::Reader<T> from(&src_tensor);
                util::feature::nhwc::View<T> into(&dst_tensor);
                ::nnfw::misc::feature::iterate(shape)
                    << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
                         const auto value = from.at(batch, ch, row, col);
                         into.at(batch, row, col, ch) = value;
                       };
                break;
              }
              case PermuteType::COPY:
              {
                const int32_t dim_0 = dst_tensor.dimension(0);
                const int32_t dim_1 = dst_tensor.dimension(1);
                const int32_t dim_2 = dst_tensor.dimension(2);
                const int32_t copy_len = dst_tensor.dimension(3);

                for (auto i = 0; i < dim_0; ++i)
                {
                  for (auto j = 0; j < dim_1; ++j)
                  {
                    for (auto k = 0; k < dim_2; ++k)
                    {
                      ir::Coordinates coords{i, j, k, 0};
                      memcpy(dst_buffer + dst_tensor.calcOffset(coords),
                             src_buffer + src_tensor.calcOffset(coords), copy_len * sizeof(T));
                    }
                  }
                }
                break;
              }
              default:
              {
                throw std::runtime_error("Unsupported Permutation");
                break;
              }
            }
            break;
          }
          default:
            throw std::runtime_error("Unsupported rank in permutation");
            break;
        }
      });
    };
    src->access(fn);
  }

  // NOTE The typeid expression is lvalue expression which refers to an object with static storage
  //      duration, of the polymorphic type const std::type_info or of some type derived from it.
  //      So std::type_info is non-copyable
  const std::type_info &underlying_type(ir::DataType type) const
  {
    switch (type)
    {
      case ir::DataType::FLOAT32:
        return typeid(float);
      case ir::DataType::INT32:
        return typeid(int32_t);
      case ir::DataType::UINT32:
        return typeid(uint32_t);
      case ir::DataType::INT64:
        return typeid(int64_t);
      case ir::DataType::BOOL8:
      case ir::DataType::QUANT_UINT8_ASYMM:
      case ir::DataType::UINT8:
        return typeid(uint8_t);
      case ir::DataType::QUANT_INT8_SYMM:
        return typeid(int8_t);
      default:
        throw std::runtime_error("IPermuteFunction: Not supported data type");
    }
  }

protected:
  std::vector<std::shared_ptr<backend::ITensor>> _src_tensors;
  std::vector<std::shared_ptr<backend::ITensor>> _dst_tensors;
  // TODO Remove this member if it is possible
  std::vector<size_t> _ranks;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_PERMUTE_FUNCTION_H__
