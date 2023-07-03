/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_TENSOR_H__
#define __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_TENSOR_H__

#include "backend/train/ITrainableTensor.h"

#include "backend/basic/Tensor.h"

namespace onert
{
namespace backend
{
namespace basic
{
namespace train
{

class TrainableTensor : public backend::train::ITrainableTensor
{
public:
  TrainableTensor() = delete;
  virtual ~TrainableTensor() = default;

public:
  TrainableTensor(const ir::OperandInfo &info, const ir::Layout layout)
    : ITrainableTensor{info}, _tensor{info, layout, nullptr}, _opt_vars{}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Set the Buffer object. This method is called for static and non-const tensor
   */
  void setBuffer(uint8_t *buffer) { _tensor.setBuffer(buffer); }

public:
  uint8_t *buffer() const override { return _tensor.buffer(); }
  /**
   * @brief Get dimension by index
   *
   * @param index Index to get diemension
   * @return size_t Dimension at index
   * @note N : dimension(0)
   *       H : dimension(1)
   *       W : dimension(2)
   *       C : dimension(3)
   */
  size_t total_size() const override { return _tensor.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    return _tensor.calcOffset(coords);
  }
  ir::Layout layout() const override { return _tensor.layout(); }
  ir::DataType data_type() const override { return _tensor.data_type(); }
  bool is_constant() const override { return _tensor.is_constant(); }
  bool is_dynamic() const override { return _tensor.is_dynamic(); }
  ir::Shape getShape() const override { return _tensor.getShape(); };
  const ir::OperandInfo &get_info() { return _tensor.get_info(); }

public:
  std::vector<ITensor *> optVars() override;
  void appendOptVar(std::unique_ptr<Tensor> opt_var) { _opt_vars.emplace_back(std::move(opt_var)); }

public:
  void fillBuffer(const std::shared_ptr<ir::Data> &data);

private:
  using ITensor::setShape;
  using ITensor::set_dynamic;
  using ITensor::applyShape;

protected:
  Tensor _tensor;
  std::vector<std::unique_ptr<Tensor>> _opt_vars; //< Optimizer variables
};

} // namespace train
} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_TENSOR_H__
