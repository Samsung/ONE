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

using DataType = luci_interpreter::DataType;

namespace
{

DataType to_internal_dtype(H5::DataType h5_type)
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
  // Only support three datatypes for now
  return DataType::Unknown;
}

template <typename T> void readTensor(H5::DataSet &tensor, T *buffer);

template <> void readTensor<float>(H5::DataSet &tensor, float *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_FLOAT);
}
template <> void readTensor<int32_t>(H5::DataSet &tensor, int32_t *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_INT);
}
template <> void readTensor<int64_t>(H5::DataSet &tensor, int64_t *buffer)
{
  tensor.read(buffer, H5::PredType::NATIVE_LONG);
}

} // namespace

namespace record_minmax
{

int32_t HDF5Importer::numInputs(int32_t rid)
{
  auto records = _value_grp.openGroup(std::to_string(rid));
  return records.getNumObjs();
}

void HDF5Importer::read(int32_t rid, int32_t iid, DataType *dtype, void *buffer)
{
  auto record = _value_grp.openGroup(std::to_string(rid));
  auto tensor = record.openDataSet(std::to_string(iid));

  auto tensor_dtype = tensor.getDataType();
  *dtype = to_internal_dtype(tensor_dtype);

  switch (*dtype)
  {
    case DataType::FLOAT32:
    {
      readTensor<float>(tensor, static_cast<float *>(buffer));
      break;
    }
    case DataType::S32:
    {
      readTensor<int32_t>(tensor, static_cast<int32_t *>(buffer));
      break;
    }
    case DataType::S64:
    {
      readTensor<int64_t>(tensor, static_cast<int64_t *>(buffer));
      break;
    }
    default:
      throw std::runtime_error{"Unsupported data type for input data (.h5)"};
  }
}

} // namespace record_minmax
