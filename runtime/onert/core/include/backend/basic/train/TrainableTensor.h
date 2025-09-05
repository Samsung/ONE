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

#include "backend/basic/Tensor.h"
#include "backend/train/ITrainableTensor.h"

namespace onert::backend::basic::train
{

class TrainableTensor : public backend::train::ITrainableTensor
{
public:
  TrainableTensor() = delete;
  virtual ~TrainableTensor() = default;

public:
  TrainableTensor(const ir::OperandInfo &info)
    : ITrainableTensor{info}, _tensor{info, nullptr}, _opt_vars{}
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

public:
  std::vector<ITensor *> optVars() override;
  void appendOptVar(std::unique_ptr<Tensor> opt_var) { _opt_vars.emplace_back(std::move(opt_var)); }
  void setOptVarBuffer(uint8_t *buffer, size_t pos) { _opt_vars.at(pos)->setBuffer(buffer); }

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

} // namespace onert::backend::basic::train

#endif // __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_TENSOR_H__
