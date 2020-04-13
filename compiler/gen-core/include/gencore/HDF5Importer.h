/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __GENCORE_HDF5IMPORTER_H__
#define __GENCORE_HDF5IMPORTER_H__

#include "HDF5Common.h"

#include <tfinfo-v2/TensorSignature.h>

#include <angkor/TensorShape.h>
#include <nncc/core/ADT/tensor/Accessor.h>

#include <H5Cpp.h>

namespace gencore
{

class HDF5Importer
{
public:
  HDF5Importer(const std::string &path) : _file{path, H5F_ACC_RDONLY}
  {
    _value_grp = _file.openGroup(value_grpname());
  }

public:
  /**
   * @brief Reads tensor data from file and store it into buf_accessor
   */
  template <typename DT>
  void read(uint32_t nth, const std::string &name, const angkor::TensorShape &shape,
            nncc::core::ADT::tensor::Accessor<DT> *buf_accessor);

private:
  H5::H5File _file;
  H5::Group _value_grp;
};

} // namespace gencore

#endif // __GENCORE_HDF5IMPORTER_H__
