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

/**
 * @file  Tensor.h
 * @brief This file contains ITensor interface, ROTensor class, and Tensor class
 */
#ifndef __ONERT_INTERP_TENSOR_H__
#define __ONERT_INTERP_TENSOR_H__

#include "Buffer.h"

#include "ir/OperandInfo.h"
#include "backend/ITensor.h"
#include "ir/Layout.h"

namespace onert
{
namespace interp
{

/**
 * @brief Interface to handle Tensor in interpreter
 */
class ITensor : public backend::ITensor
{
public:
  virtual ~ITensor() = default;

public:
  virtual uint8_t *buffer() const = 0;
  /**
   * @brief   Return shared pointer for buffer
   * @return  Buffer shared pointer
   */
  virtual std::shared_ptr<const Buffer> shareBuffer() const = 0;
  /**
   * @brief   Return read-only buffer pointer
   * @return  Read-only buffer pointer
   */
  virtual const uint8_t *bufferRO() const = 0;
  /**
   * @brief   Return shared pointer for data
   * @return  Data shared pointer
   */
  virtual std::shared_ptr<const ir::Data> shareData() const = 0;
  /**
   * @brief     Set internal/external buffer
   * @param[in] buffer  Buffer pointer
   */
  virtual void setBuffer(std::shared_ptr<const Buffer> buffer) = 0;
  /**
   * @brief     Set data reference (including constant, input)
   * @param[in] data  Data pointer
   */
  virtual void setData(std::shared_ptr<const ir::Data> data) = 0;
  virtual void releaseData() = 0;

  virtual size_t total_size() const = 0;
  virtual size_t calcOffset(const ir::Coordinates &coords) const = 0;

  virtual bool has_padding() const = 0;
  /**
   * @brief   Return data type of tensor
   * @return  Data type of tensor
   */
  virtual ir::DataType data_type() const = 0;
  /**
   * @brief   Return TensorInfo
   * @return  TensorInfo
   */
  virtual const ir::OperandInfo &tensorInfo() const = 0;
  /**
   * @brief   Return number of elements
   * @return  Number of elements
   */
  virtual uint64_t num_elements() const = 0;
  void access(const std::function<void(backend::ITensor &tensor)> &fn) final;
};

/**
 * @brief Class to handle tensor in interpreter as read-only
 */
class ROTensor final : public ITensor
{
public:
  ROTensor() = delete;
  ROTensor(const ir::OperandInfo &info) : _info(info)
  {
    // DO NOTHING
  }

public:
  uint8_t *buffer() const override { throw std::runtime_error{"Read only tensor"}; }
  std::shared_ptr<const Buffer> shareBuffer() const override
  {
    throw std::runtime_error{"Read only tensor"};
  }
  const uint8_t *bufferRO() const override { return _data->base(); }
  std::shared_ptr<const ir::Data> shareData() const override { return _data; }
  void setBuffer(std::shared_ptr<const Buffer> buffer) override { _data = buffer; }
  void setData(std::shared_ptr<const ir::Data> data) override { _data = data; }
  void releaseData() override { _data = nullptr; }

  size_t total_size() const override { return _info.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override;
  ir::Layout layout() const override;
  bool is_dynamic() const override { return false; }
  bool has_padding() const override { return false; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const override { return _info.typeInfo().scale(); }
  int32_t data_zero_point() const override { return _info.typeInfo().zero_point(); }
  const ir::OperandInfo &tensorInfo() const override { return _info; }
  uint64_t num_elements() const override { return _info.shape().num_elements(); };
  ir::Shape getShape() const override;

private:
  const ir::OperandInfo _info;
  std::shared_ptr<const ir::Data> _data{nullptr};
};

/**
 * @brief Class to handle tensor in interpreter as writable
 */
class Tensor final : public ITensor
{
public:
  Tensor() = delete;
  Tensor(const ir::OperandInfo &info) : _info(info)
  {
    // DO NOTHING
  }

public:
  uint8_t *buffer() const override { return _buffer->baseWritable(); }
  std::shared_ptr<const Buffer> shareBuffer() const override { return _buffer; };
  const uint8_t *bufferRO() const override { return _buffer->base(); }
  std::shared_ptr<const ir::Data> shareData() const override { return _buffer; }
  void setBuffer(std::shared_ptr<const Buffer> buffer) override { _buffer = buffer; }
  void setData(std::shared_ptr<const ir::Data>) override
  {
    throw std::runtime_error{"Passed data may read-only"};
  }
  void releaseData() override { _buffer = nullptr; }

  size_t total_size() const override { return _info.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override;
  ir::Layout layout() const override;
  bool is_dynamic() const override { return false; }
  bool has_padding() const override { return false; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const override { return _info.typeInfo().scale(); }
  int32_t data_zero_point() const override { return _info.typeInfo().zero_point(); }
  const ir::OperandInfo &tensorInfo() const override { return _info; }
  uint64_t num_elements() const override { return _info.shape().num_elements(); };
  ir::Shape getShape() const override;

private:
  const ir::OperandInfo _info;
  std::shared_ptr<const Buffer> _buffer{nullptr};
};

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_TENSOR_H__
