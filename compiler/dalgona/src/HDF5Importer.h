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

#ifndef __DALGONA_HDF5IMPORTER_H__
#define __DALGONA_HDF5IMPORTER_H__

#include <luci_interpreter/core/Tensor.h>

#include <H5Cpp.h>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace dalgona
{

// HDF5Importer reads an input data saved in the hdf5 file in the given path
// The hierarchy of the hdf5 file is as follows.
// Group "/"
//  > Group "value"
//    > Group <record_idx>
//      > Dataset <input_idx>
// record_idx : index of the record (dataset file can contain multiple records)
// input_idx : index of the input (DNN model can have multiple inputs)
// Ex: the j'th input of the i'th record can be accessed by "/value/i/j"
class HDF5Importer
{
public:
  explicit HDF5Importer(const std::string &path) : _file{path, H5F_ACC_RDONLY}
  {
    // Do nothing
  }

public:
  /**
   * @brief Read tensor data from file and store it into buffer
   * @details A tensor in the file can be retrieved with (record_idx, input_idx)
   * @param record_idx : index of the record
   * @param input_idx : index of the input
   * @param dtype : pointer to write the tensor's data type
   * @param shape : pointer to write the tensor's shape
   * @param buffer : pointer to write the tensor's data
   */
  void readTensor(int32_t record_idx, int32_t input_idx, DataType *dtype, Shape *shape,
                  void *buffer);

  // Read a raw tensor (no type/shape is specified)
  void readTensor(int32_t record_idx, int32_t input_idx, void *buffer);

  bool isRawData() { return _file.openGroup("value").attrExists("rawData"); }

  int32_t numRecords() { return _file.openGroup("value").getNumObjs(); }

  int32_t numInputs(int32_t record_idx)
  {
    return _file.openGroup("value/" + std::to_string(record_idx)).getNumObjs();
  }

private:
  H5::H5File _file;
};

} // namespace dalgona

#endif // __DALGONA_HDF5IMPORTER_H__
