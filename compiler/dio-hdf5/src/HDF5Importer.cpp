/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dio_hdf5/HDF5Importer.h"

#include <H5Cpp.h>

#include <string>
#include <vector>
#include <cassert>
#include <stdexcept>

using Shape = std::vector<loco::Dimension>;
using DataType = loco::DataType;

namespace
{

Shape toInternalShape(const H5::DataSpace &dataspace)
{
  int rank = dataspace.getSimpleExtentNdims();

  std::vector<hsize_t> dims;
  dims.resize(rank, 0);
  dataspace.getSimpleExtentDims(dims.data());

  Shape res;
  for (int axis = 0; axis < rank; ++axis)
  {
    res.emplace_back(dims[axis]);
  }

  return res;
}

DataType toInternalDtype(const H5::DataType &h5_type)
{
  if (h5_type == H5::PredType::IEEE_F32BE || h5_type == H5::PredType::IEEE_F32LE)
  {
    return DataType::FLOAT32;
  }
  if (h5_type == H5::PredType::STD_I32BE || h5_type == H5::PredType::STD_I32LE)
  {
    return DataType::S32;
  }
  if (h5_type == H5::PredType::STD_I64BE || h5_type == H5::PredType::STD_I64LE)
  {
    return DataType::S64;
  }
  if (h5_type.getClass() == H5T_class_t::H5T_ENUM)
  {
    // We follow the numpy format
    // In numpy 1.19.0, np.bool_ is saved as H5T_ENUM
    // - (name, value) -> (FALSE, 0) and (TRUE, 1)
    // - value dtype is H5T_STD_I8LE
    // TODO Find a general way to recognize BOOL type
    char name[10];
    int8_t value[2] = {0, 1};
    if (H5Tenum_nameof(h5_type.getId(), value, name, 10) < 0)
      return DataType::Unknown;

    if (std::string(name) != "FALSE")
      return DataType::Unknown;

    if (H5Tenum_nameof(h5_type.getId(), value + 1, name, 10) < 0)
      return DataType::Unknown;

    if (std::string(name) != "TRUE")
      return DataType::Unknown;

    return DataType::BOOL;
  }
  // TODO Support more datatypes
  return DataType::Unknown;
}

void readTensorData(H5::DataSet &tensor, uint8_t *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_UINT8);
}

void readTensorData(H5::DataSet &tensor, float *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_FLOAT);
}

void readTensorData(H5::DataSet &tensor, int32_t *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_INT);
}

void readTensorData(H5::DataSet &tensor, int64_t *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_LONG);
}

} // namespace

namespace dio
{
namespace hdf5
{

HDF5Importer::HDF5Importer(const std::string &path)
{
  if (_file.isHdf5(path) == false)
    throw std::runtime_error("Given data file is not HDF5");

  _file = H5::H5File(path, H5F_ACC_RDONLY);
}

int32_t HDF5Importer::numInputs(int32_t record_idx) const
{
  auto records = _group.openGroup(std::to_string(record_idx));
  return records.getNumObjs();
}

void HDF5Importer::readTensor(int32_t record_idx, int32_t input_idx, void *buffer,
                              size_t buffer_bytes) const
{
  auto record = _group.openGroup(std::to_string(record_idx));
  auto tensor = record.openDataSet(std::to_string(input_idx));

  if (tensor.getInMemDataSize() != buffer_bytes)
    throw std::runtime_error("Buffer size does not match with the size of tensor data");

  readTensorData(tensor, static_cast<uint8_t *>(buffer));
}

void HDF5Importer::readTensor(int32_t record_idx, int32_t input_idx, DataType *dtype, Shape *shape,
                              void *buffer, size_t buffer_bytes) const
{
  auto record = _group.openGroup(std::to_string(record_idx));
  auto tensor = record.openDataSet(std::to_string(input_idx));

  auto tensor_dtype = tensor.getDataType();
  *dtype = toInternalDtype(tensor_dtype);

  auto tensor_shape = tensor.getSpace();
  *shape = toInternalShape(tensor_shape);

  if (tensor.getInMemDataSize() != buffer_bytes)
    throw std::runtime_error("Buffer size does not match with the size of tensor data");

  switch (*dtype)
  {
    case DataType::FLOAT32:
      readTensorData(tensor, static_cast<float *>(buffer));
      break;
    case DataType::S32:
      readTensorData(tensor, static_cast<int32_t *>(buffer));
      break;
    case DataType::S64:
      readTensorData(tensor, static_cast<int64_t *>(buffer));
      break;
    case DataType::BOOL:
      readTensorData(tensor, static_cast<uint8_t *>(buffer));
      break;
    default:
      throw std::runtime_error{"Unsupported data type for input data (.h5)"};
  }
}

} // namespace hdf5
} // namespace dio
