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

#include "gencore/HDF5Exporter.h"

#include <angkor/TensorShape.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Reader.h>

#include <H5Cpp.h>

namespace
{

template <typename DT> H5::PredType get_h5_datatype();

template <> H5::PredType get_h5_datatype<float>() { return H5::PredType::NATIVE_FLOAT; }

template <typename DT> H5::PredType get_h5_store_format();

template <> H5::PredType get_h5_store_format<float>() { return H5::PredType::IEEE_F32BE; }

} // namespace

namespace gencore
{

template <typename DT>
void H5Exporter::write(uint32_t nth, const std::string &name, const angkor::TensorShape &shape,
                       const nncc::core::ADT::tensor::Reader<DT> &buf_reader)
{
  // Record tensor values
  {
    const auto rank = shape.rank();

    hsize_t dims[rank];

    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      dims[axis] = shape.dim(axis);
    }

    H5::DataSpace dataspace(rank, dims);

    auto dataset =
        _value_grp.createDataSet(value_filename(nth), get_h5_store_format<DT>(), dataspace);

    DT *h5_data = new DT[nncc::core::ADT::tensor::num_elements(shape)];
    {
      using nncc::core::ADT::tensor::IndexEnumerator;
      using nncc::core::ADT::tensor::LexicalLayout;

      LexicalLayout layout{};
      for (IndexEnumerator e{shape}; e.valid(); e.advance())
      {
        auto i = e.current();
        h5_data[layout.offset(shape, i)] = buf_reader.at(i);
      }
    }

    dataset.write(h5_data, get_h5_datatype<DT>());

    delete[] h5_data;
  }

  // Record name
  {
    H5::DataSpace name_dataspace(H5S_SCALAR);
    H5::StrType name_datatype(H5::PredType::C_S1, name.size());

    auto name_attr = _name_grp.createAttribute(value_filename(nth), name_datatype, name_dataspace);

    name_attr.write(name_datatype, name);
  }
}

// template instantiation
template void H5Exporter::write<float>(uint32_t, const std::string &, const angkor::TensorShape &,
                                       const nncc::core::ADT::tensor::Reader<float> &);

} // namespace gencore
