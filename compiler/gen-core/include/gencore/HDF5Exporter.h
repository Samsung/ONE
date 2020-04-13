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

#ifndef __GENCORE_HDF5EXPORTER_H__
#define __GENCORE_HDF5EXPORTER_H__

#include "HDF5Common.h"

#include <angkor/TensorShape.h>
#include <nncc/core/ADT/tensor/Reader.h>

#include <H5Cpp.h>

namespace gencore
{

class H5Exporter
{
public:
  H5Exporter(const std::string &path) : _file{path.c_str(), H5F_ACC_TRUNC}
  {
    _value_grp = _file.createGroup(value_grpname());
    _name_grp = _file.createGroup(name_grpname());
  }

public:
  template <typename DT>
  void write(uint32_t nth, const std::string &name, const angkor::TensorShape &shape,
             const nncc::core::ADT::tensor::Reader<DT> &buf_reader);

private:
  H5::H5File _file;
  H5::Group _value_grp;
  H5::Group _name_grp;
};

} // namespace gencore

#endif // __GENCORE_HDF5EXPORTER_H__
