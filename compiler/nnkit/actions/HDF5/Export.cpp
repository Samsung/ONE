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

#include "Common.h"

#include <nnkit/Action.h>

#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <H5Cpp.h>

using nnkit::TensorContext;

class HD5ExportAction final : public nnkit::Action
{
public:
  HD5ExportAction(const std::string &path) : _file{path, H5F_ACC_TRUNC}
  {
    _value_grp = _file.createGroup(value_grpname());
    _name_grp = _file.createGroup(name_grpname());
  }

public:
  void run(TensorContext &ctx) override
  {
    for (uint32_t n = 0; n < ctx.size(); ++n)
    {
      using nncc::core::ADT::tensor::Reader;

      // TODO Support other data types
      auto fn = [this](const TensorContext &ctx, uint32_t n, const Reader<float> &t) {
        const auto name = ctx.name(n);
        const auto shape = ctx.shape(n);

        const auto rank = shape.rank();

        hsize_t dims[rank];

        for (uint32_t axis = 0; axis < rank; ++axis)
        {
          dims[axis] = shape.dim(axis);
        }

        H5::DataSpace dataspace(rank, dims);

        auto dataset =
          _value_grp.createDataSet(value_filename(n), H5::PredType::IEEE_F32BE, dataspace);

        float *data = new float[nncc::core::ADT::tensor::num_elements(shape)];

        using nncc::core::ADT::tensor::Index;
        using nncc::core::ADT::tensor::IndexEnumerator;
        using nncc::core::ADT::tensor::LexicalLayout;

        LexicalLayout layout{};

        for (IndexEnumerator e{shape}; e.valid(); e.advance())
        {
          auto i = e.current();
          data[layout.offset(shape, i)] = t.at(i);
        }

        dataset.write(data, H5::PredType::NATIVE_FLOAT);

        delete[] data;

        // Record name
        {
          H5::DataSpace name_dataspace(H5S_SCALAR);
          H5::StrType name_datatype(H5::PredType::C_S1, name.size());

          auto name_attr =
            _name_grp.createAttribute(value_filename(n), name_datatype, name_dataspace);

          name_attr.write(name_datatype, name);
        }
      };

      ctx.getConstFloatTensor(n, fn);
    }
  }

private:
  H5::H5File _file;
  H5::Group _value_grp;
  H5::Group _name_grp;
};

#include <nnkit/CmdlineArguments.h>
#include <stdex/Memory.h>

extern "C" std::unique_ptr<nnkit::Action> make_action(const nnkit::CmdlineArguments &args)
{
  return stdex::make_unique<HD5ExportAction>(args.at(0));
}
