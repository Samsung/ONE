/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_ICONSTANT_INITIALIZER_H__
#define __ONERT_BACKEND_ICONSTANT_INITIALIZER_H__

#include <unordered_map>
#include <functional>

#include "ITensorBuilder.h"
#include "ir/Coordinates.h"
#include "ir/Layout.h"
#include "ir/Operand.h"
#include "ir/Operands.h"
#include "ir/OperationVisitor.h"
#include "ir/OpSequence.h"
#include "util/logging.h"

namespace
{
template <typename T>
static void Init(const onert::ir::Operand &model_obj, onert::backend::ITensor &obj, const bool copy,
                 const onert::ir::Layout frontend_layout = onert::ir::Layout::UNKNOWN)
{
  const auto shape = model_obj.shape();
  assert(model_obj.data());
  auto base = reinterpret_cast<const T *>(model_obj.data()->base());

  obj.access([&](::onert::backend::ITensor &tensor) {
    switch (shape.rank())
    {
      case 0:
      {
        assert(model_obj.data()->size() == sizeof(T));
        const auto value = *reinterpret_cast<const T *>(base);
        T *into = reinterpret_cast<T *>(tensor.buffer());
        *into = value;
        break;
      }
      case 1:
      {
        auto vec_size = shape.dim(0);
        for (int32_t n = 0; n < vec_size; ++n)
        {
          const T *from = reinterpret_cast<const T *>(base) + n;
          const auto value = *from;

          T *into = reinterpret_cast<T *>(tensor.buffer()) + n;

          *into = value;
        }
        break;
      }
      case 2:
      {
        const int32_t copy_len = shape.dim(1);

        for (auto i = 0; i < shape.dim(0); ++i)
        {
          ::onert::ir::Coordinates coords{i, 0};
          memcpy(tensor.buffer() + tensor.calcOffset(coords), base + i * copy_len,
                 copy_len * sizeof(T));
        }
        break;
      }
      case 3:
      {
        const int32_t width = shape.dim(1);
        const int32_t copy_len = shape.dim(2);

        for (auto i = 0; i < shape.dim(0); ++i)
        {
          for (auto j = 0; j < shape.dim(1); ++j)
          {
            ::onert::ir::Coordinates coords{i, j, 0};
            memcpy(tensor.buffer() + tensor.calcOffset(coords),
                   base + i * width * copy_len + j * copy_len, copy_len * sizeof(T));
          }
        }
        break;
      }
      case 4:
      {
        const int32_t height = shape.dim(1);
        const int32_t width = shape.dim(2);
        const int32_t copy_len = shape.dim(3);
        for (auto i = 0; i < shape.dim(0); ++i)
        {
          for (auto j = 0; j < shape.dim(1); ++j)
          {
            for (auto k = 0; k < shape.dim(2); ++k)
            {
              if (copy)
              {
                ::onert::ir::Coordinates coords{i, j, k, 0};
                memcpy(tensor.buffer() + tensor.calcOffset(coords),
                       base + i * height * width * copy_len + j * width * copy_len + k * copy_len,
                       copy_len * sizeof(T));
              }
              else
              {
                for (auto l = 0; l < shape.dim(3); ++l)
                {
                  const auto coords = ::onert::ir::convertCoordinates({i, j, k, l}, frontend_layout,
                                                                      tensor.layout());
                  T *into = reinterpret_cast<T *>(tensor.buffer() + tensor.calcOffset(coords));
                  T value = *(base + i * height * width * copy_len + j * width * copy_len +
                              k * copy_len + l);
                  *into = value;
                }
              }
            }
          }
        }
        break;
      }
      default:
        throw std::runtime_error{"Not yet supported"};
    }
  });
}

template <typename T>
void copyInit(const onert::ir::Operand &model_obj, onert::backend::ITensor &obj)
{
  Init<T>(model_obj, obj, true);
}

template <typename T>
void permuteInit(const onert::ir::Operand &model_obj, onert::backend::ITensor &obj,
                 const onert::ir::Layout frontend_layout)
{
  const bool copy = frontend_layout == obj.layout();
  Init<T>(model_obj, obj, copy, frontend_layout);
}

} // namespace

namespace onert
{
namespace backend
{

class IConstantInitializer : public ir::OperationVisitor
{
public:
  virtual ~IConstantInitializer() = default;

public:
  void run()
  {
    assert(tensor_builder().get());
    for (const auto &it : _init_map)
    {
      const auto &ind = it.first;
      const auto &fn = it.second;

      const auto &model_obj = _operands.at(ind);
      auto tensor_obj = tensor_builder()->tensorAt(ind);
      assert(tensor_obj != nullptr);
      fn(model_obj, *tensor_obj);
      VERBOSE(FillOperandData) << "Fill data for operand " << ind.value() << std::endl;
    }
    _init_map.clear();
  }

public:
  IConstantInitializer(const ir::Operands &operands)
      : _operands{operands}, _current_op_seq_layout{ir::Layout::UNKNOWN}
  {
  }

public:
  using Initializer = std::function<void(const ir::Operand &, backend::ITensor &)>;

  void setLayout(ir::Layout layout) { _current_op_seq_layout = layout; }

protected:
  using OperationVisitor::visit;

protected:
  virtual std::shared_ptr<ITensorBuilder> tensor_builder() const = 0;

public:
  virtual void registerDefaultInitializer(const ir::OperandIndex &index, const ir::Operand &obj)
  {
    // use `registerPermuteInitializer` as default
    registerPermuteInitializer(index, obj);
  }

public:
  void registerCopyInitializer(const ir::OperandIndex &index, const ir::Operand &obj);
  void registerPermuteInitializer(const ir::OperandIndex &index, const ir::Operand &obj);

public:
  void registerCustomInitializer(const ir::OperandIndex &index, const ir::Operand &obj,
                                 void (*customInit)(const onert::ir::Operand &model_obj,
                                                    onert::backend::ITensor &obj))
  {
    // For only CONSTANTS
    // TODO Add to check if tensor has been allocated
    if (!obj.isConstant())
      return;

    using namespace std::placeholders;
    _init_map[index] = std::bind(customInit, _1, _2);
  }

public:
  bool exist(const ir::OperandIndex &ind) { return _init_map.find(ind) != _init_map.end(); }

protected:
  const ir::Operands &_operands;
  std::unordered_map<ir::OperandIndex, Initializer> _init_map;
  ir::Layout _current_op_seq_layout; // TODO Rename this to _current_layout
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ICONSTANT_INITIALIZER_H__
