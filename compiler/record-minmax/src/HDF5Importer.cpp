/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "HDF5Importer.h"

#include <H5Cpp.h>

#include <string>
#include <cassert>
#include <stdexcept>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace
{

Shape toInternalShape(const H5::DataSpace &dataspace)
{
  int rank = dataspace.getSimpleExtentNdims();

  std::vector<hsize_t> dims;
  dims.resize(rank, 0);
  dataspace.getSimpleExtentDims(dims.data());

  Shape res(rank);
  for (int axis = 0; axis < rank; ++axis)
  {
    res.dim(axis) = dims[axis];
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
    // In numpy 1.19.0, np.bool_ is saved as H5T_ENUM, where FALSE = 0 and TRUE = 1 (H5T_STD_I8LE)
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
  // Only support three datatypes for now
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

namespace record_minmax
{

int32_t HDF5Importer::numInputs(int32_t record_idx)
{
  auto records = _value_grp.openGroup(std::to_string(record_idx));
  return records.getNumObjs();
}

void HDF5Importer::readTensor(int32_t record_idx, int32_t input_idx, void *buffer)
{
  auto record = _value_grp.openGroup(std::to_string(record_idx));
  auto tensor = record.openDataSet(std::to_string(input_idx));

  readTensorData(tensor, static_cast<uint8_t *>(buffer));
}

void HDF5Importer::readTensor(int32_t record_idx, int32_t input_idx, DataType *dtype, Shape *shape,
                              void *buffer)
{
  auto record = _value_grp.openGroup(std::to_string(record_idx));
  auto tensor = record.openDataSet(std::to_string(input_idx));

  auto tensor_dtype = tensor.getDataType();
  *dtype = toInternalDtype(tensor_dtype);

  auto tensor_shape = tensor.getSpace();
  *shape = toInternalShape(tensor_shape);

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

} // namespace record_minmax
