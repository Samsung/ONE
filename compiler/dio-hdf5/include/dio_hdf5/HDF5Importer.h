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

#ifndef __DIO_HDF5_H__
#define __DIO_HDF5_H__

#include <H5Cpp.h>

#include <loco.h>

#include <string>
#include <vector>

namespace dio
{
namespace hdf5
{

// HDF5Importer reads an input data saved in the hdf5 file in the given path
// The hierarchy of the hdf5 file is as follows.
// Group "/"
//  > Group <group_name>
//    > Group <data_idx>
//      > Dataset <input_idx>
// data_idx : index of the data (dataset file can contain multiple data)
// input_idx : index of the input (DNN model can have multiple inputs)
// Ex: the j'th input of the i'th data of group 'value' can be accessed by "/value/i/j"
class HDF5Importer final
{
public:
  explicit HDF5Importer(const std::string &path);

public:
  /**
   * @note importGroup has to be called before readTensor is called
   *        Otherwise, readTensor will throw an exception
   */
  void importGroup(const std::string &group) { _group = _file.openGroup(group); }

  /**
   * @brief Read tensor data from file and store it into buffer
   * @details A tensor in the file can be retrieved with (data_idx, input_idx)
   * @param data_idx : index of the data
   * @param input_idx : index of the input
   * @param dtype : pointer to write the tensor's data type
   * @param shape : pointer to write the tensor's shape
   * @param buffer : pointer to write the tensor's data
   * @param buffer_bytes : byte size of the buffer
   */
  void readTensor(int32_t data_idx, int32_t input_idx, loco::DataType *dtype,
                  std::vector<loco::Dimension> *shape, void *buffer, size_t buffer_bytes) const;

  // Read a raw tensor (no type/shape is specified)
  void readTensor(int32_t data_idx, int32_t input_idx, void *buffer, size_t buffer_bytes) const;

  bool isRawData() const { return _group.attrExists("rawData"); }

  int32_t numData() const { return _group.getNumObjs(); }

  int32_t numInputs(int32_t data_idx) const;

private:
  H5::H5File _file;
  H5::Group _group;
};

} // namespace hdf5
} // namespace dio

#endif // __DIO_HDF5_H__
