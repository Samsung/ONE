
/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "backend/ITensor.h"

template <typename T> class MockTensor : public onert::backend::ITensor
{
public:
  MockTensor<T>(onert::ir::Shape &shape, T *buf, onert::ir::Layout layout)
    : _buf(reinterpret_cast<uint8_t *>(buf)), _shape(shape), _layout(layout)
  {
  }

public:
  uint8_t *buffer() const override { return _buf; }

  size_t calcOffset(const onert::ir::Coordinates &coords) const override
  {
    size_t rank = _shape.rank();
    rank = rank == 0 ? 1 : rank;
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i)
    {
      auto dim = _shape.rank() == 0 ? 1 : _shape.dim(i);
      offset = offset * dim + coords[i];
    }
    offset *= sizeof(T);

    return offset;
  }

  onert::ir::Shape getShape() const override { return _shape; }

public: // DUMMY methods
  size_t total_size() const override { return 0; }
  onert::ir::Layout layout() const override { return _layout; }
  onert::ir::DataType data_type() const override { return onert::ir::DataType::UINT8; }
  float data_scale() const override { return 0; }
  int32_t data_zero_point() const override { return 0; }
  const std::vector<float> &data_scales() const override { return _dummy_scales; }
  const std::vector<int32_t> &data_zero_points() const override { return _dummy_zerops; }
  bool has_padding() const override { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) override {}
  bool is_dynamic() const override { return false; }

private:
  uint8_t *_buf = nullptr;
  onert::ir::Shape _shape;
  onert::ir::Layout _layout = onert::ir::Layout::UNKNOWN;
  std::vector<float> _dummy_scales;
  std::vector<int32_t> _dummy_zerops;
};
