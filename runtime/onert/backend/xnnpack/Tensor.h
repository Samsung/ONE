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

#ifndef __ONERT_BACKEND_XNNPACK_TENSOR_H__
#define __ONERT_BACKEND_XNNPACK_TENSOR_H__

#include <backend/cpu_common/Tensor.h>
#include <ir/Data.h>

namespace onert
{
namespace backend
{
namespace xnnpack
{

using Tensor = cpu_common::Tensor;

/**
 * @brief Class that uses data from external memory that is not managed by a backend
 *        instead of allocating and copying the data. ExternalTensor's data pointer points to
 *        an address of memory such as where memory is already allocated, or mmapped area.
 *        This is meaning that ExternalTensor can take all of types' ir::Data.
 *        To support this, assume below things no padding, always NHWC layout,
 *        constant tensor and not dynamic.
 */
class ExternalTensor : public Tensor
{
public:
  ExternalTensor() = delete;
  virtual ~ExternalTensor();

public:
  ExternalTensor(const ir::OperandInfo &info, const ir::Layout layout)
      : Tensor(info, layout, nullptr)
  {
    assert(_layout == ir::Layout::NHWC);
    assert(_info.isConstant());
    assert(_info.isDynamic() == false);
  }

public:
  /**
   * @brief     set Data to be shared from external so that this ExternalTensor will not be
   *            allocated on XNNPACK backend
   * @param[in] data    data of Operand to be set
   */
  void setData(const std::shared_ptr<ir::Data> data)
  {
    assert(data != nullptr);
    _data = data;
    // Note. Some op such as cker::Conv could take buffer as nullptr.
    // That's why _buffer also would be used
    _buffer = const_cast<uint8_t *>(_data->base());
  }

public:
  uint8_t *buffer() const override { return _buffer; }

  bool is_constant() const override { return true; }
  bool is_dynamic() const override { return false; }
  void set_dynamic() override
  {
    throw std::runtime_error("This tensor does not support changing dynamic");
  }

  void setShape(const ir::Shape &) override
  {
    throw std::runtime_error("This tensor does not support changing shape");
  }

  void increase_ref() override { ++_num_references; }

  void decrease_ref() override
  {
    assert(_data != nullptr);
    assert(_num_references > 0);
    --_num_references;
    if (_num_references == 0)
    {
      _data.reset();
      _buffer = nullptr;
    }
  }

  /**
   * @brief Reset reference count to zero and release data
   */
  void reset_ref() override
  {
    assert(_data != nullptr);
    assert(_num_references > 0);
    _num_references = 0;

    _data.reset();
    _buffer = nullptr;
  }

  int32_t num_references() override { return _num_references; }

private:
  std::shared_ptr<const ir::Data> _data;
};

} // namespace xnnpack
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_XNNPACK_TENSOR_H__
