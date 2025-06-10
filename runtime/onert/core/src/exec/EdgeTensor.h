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

#ifndef __ONERT_EXEC_EDGE_TENSOR_H__
#define __ONERT_EXEC_EDGE_TENSOR_H__

#include "backend/IPortableTensor.h"

#include <memory>

namespace onert::exec
{

class EdgeTensor : public backend::IPortableTensor
{
public:
  EdgeTensor(const ir::OperandInfo &info, ir::Layout layout)
    : IPortableTensor(info), _layout{layout}, _buffer{nullptr}, _ref_count{0}
  {
  }
  ~EdgeTensor() = default;

  uint8_t *buffer() const override { return _buffer.get(); }
  ir::Layout layout() const { return _layout; }
  void set_dynamic() override { _info.setDynamic(); }
  bool applyShape(const ir::Shape &new_shape) override;
  void setShape(const ir::Shape &new_shape) override { _info.shape(new_shape); }

  void allocate_buffer()
  {
    const auto total_size = _info.total_size();
    _buffer = std::make_unique<uint8_t[]>(total_size);
    _ref_count = 1;
  }

  void increase_ref() { _ref_count++; }

  void decrease_ref()
  {
    assert(_ref_count > 0);
    _ref_count--;
    if (_ref_count == 0)
    {
      _buffer.reset();
    }
  }

private:
  ir::Layout _layout;
  std::unique_ptr<uint8_t[]> _buffer;
  int32_t _ref_count;
};

} // namespace onert::exec

#endif // __ONERT_EXEC_EDGE_TENSOR_H__
