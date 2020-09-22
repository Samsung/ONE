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

#include "feature/IndexIterator.h"
#include "feature/nchw/Reader.h"
#include "feature/nchw/View.h"
#include "feature/nhwc/Reader.h"
#include "feature/nhwc/View.h"

#include "backend/ITensor.h"
#include "exec/IFunction.h"
#include "ir/Index.h"
#include "ir/Shape.h"
#include <memory>
#include <typeinfo>
#include "util/Utils.h"
#include <vector>

namespace onert
{
namespace exec
{

class IPermuteFunction : public IFunction
{
protected:
  enum class PermuteType
  {
    NHWC_TO_NCHW,
    NCHW_TO_NHWC,
    COPY
  };

public:
  virtual void run() override
  {
    // TODO Optimization : Make control does not reach here? when (_src_tensors.size() == 0)
    assert(_src_tensors.size() == _dst_tensors.size());
    auto src_it = _src_tensors.begin();
    auto dst_it = _dst_tensors.begin();
    while (src_it != _src_tensors.end())
    {
      auto src_tensor = *src_it;
      auto dst_tensor = *dst_it;
      if (src_tensor != dst_tensor)
      {
        const auto rank = src_tensor->num_dimensions();
        permute(src_tensor, dst_tensor, rank);
      }
      src_it++;
      dst_it++;
    }
  }

  virtual void prepare() override { optimize(); }

  virtual void optimize() = 0;

protected:
  void permute(backend::ITensor *src_tensor, backend::ITensor *dst_tensor, size_t rank)
  {
    assert(src_tensor != dst_tensor);
    assert(underlying_type(src_tensor->data_type()) == underlying_type(dst_tensor->data_type()));
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
      case ir::DataType::QUANT_INT8_ASYMM:
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

private:
  // TODO make src const by proving const access()
  template <class T> void permute(backend::ITensor *src, backend::ITensor *dst, size_t rank)
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
        if (rank == 4 && permute_type != PermuteType::COPY)
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
              const feature::nhwc::Reader<T> from(&src_tensor);
              feature::nchw::View<T> into(&dst_tensor);
              feature::iterate(shape)
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
              const feature::nchw::Reader<T> from(&src_tensor);
              feature::nhwc::View<T> into(&dst_tensor);
              feature::iterate(shape)
                  << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
                       const auto value = from.at(batch, ch, row, col);
                       into.at(batch, row, col, ch) = value;
                     };
              break;
            }
            default:
            {
              throw std::runtime_error("Unsupported Permutation");
              break;
            }
          }
        }
        else if (!src_tensor.has_padding() && !dst_tensor.has_padding())
        {
          auto src_size = src_tensor.total_size();
          assert(src_size <= dst_tensor.total_size());
          memcpy(dst_tensor.buffer(), src_tensor.buffer(), src_size);
        }
        else
        {
          auto loop_shape = src_tensor.getShape();
          const auto copy_axis = loop_shape.rank() - 1;
          const auto copy_len = loop_shape.dim(copy_axis) * sizeof(T);
          loop_shape.dim(copy_axis) = 1;
          ShapeLoop(loop_shape, [&](const onert::ir::Coordinates &coords) {
            memcpy(dst_tensor.buffer() + dst_tensor.calcOffset(coords),
                   src_tensor.buffer() + src_tensor.calcOffset(coords), copy_len);
          });
        }
      });
    };
    src->access(fn);
  }

protected:
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
      case ir::DataType::QUANT_INT8_ASYMM:
      case ir::DataType::QUANT_INT8_SYMM:
        return typeid(int8_t);
      default:
        throw std::runtime_error("IPermuteFunction: Not supported data type");
    }
  }

protected:
  std::vector<backend::ITensor *> _src_tensors;
  std::vector<backend::ITensor *> _dst_tensors;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_PERMUTE_FUNCTION_H__
