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

#include "gencore/HDF5Importer.h"
#include "gencore/HDF5Common.h"

#include <angkor/TensorShape.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Overlay.h>
#include <nncc/core/ADT/tensor/Accessor.h>

#include <H5Cpp.h>

#include <cassert>

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
void HDF5Importer::read(uint32_t nth, const std::string &name, const angkor::TensorShape &shape,
                        nncc::core::ADT::tensor::Accessor<DT> *buf_accessor)
{
  assert(buf_accessor != nullptr);

  try
  {
    auto dataset = _value_grp.openDataSet(value_filename(nth));

    assert(dataset.getDataType() == get_h5_store_format<DT>());

    std::vector<DT> file_buf;
    {
      file_buf.resize(nncc::core::ADT::tensor::num_elements(shape));
      dataset.read(file_buf.data(), get_h5_datatype<DT>());
    }

    using nncc::core::ADT::tensor::IndexEnumerator;
    using nncc::core::ADT::tensor::LexicalLayout;

    LexicalLayout layout{};

    for (IndexEnumerator e{shape}; e.valid(); e.advance())
    {
      auto i = e.current();
      buf_accessor->at(i) = file_buf[layout.offset(shape, i)];
    }
  }
  catch (const H5::FileIException &)
  {
    // Skip if data is not present in HDF5 file
  }
}

// template instantiation
template void HDF5Importer::read<float>(uint32_t, const std::string &, const angkor::TensorShape &,
                                        nncc::core::ADT::tensor::Accessor<float> *);

} // namespace gencore
