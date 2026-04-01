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

#include <cassert>

using nnkit::TensorContext;

class HD5ImportAction final : public nnkit::Action
{
public:
  HD5ImportAction(const std::string &path) : _file{path, H5F_ACC_RDONLY}
  {
    _value_grp = _file.openGroup(value_grpname());
  }

public:
  void run(TensorContext &ctx) override
  {
    for (uint32_t n = 0; n < ctx.size(); ++n)
    {
      using nncc::core::ADT::tensor::Accessor;

      auto fn = [this](const TensorContext &ctx, uint32_t n, Accessor<float> &t) {
        const auto name = ctx.name(n);

        auto dataset = _value_grp.openDataSet(value_filename(n));

        // TODO Support non-float tensors
        assert(dataset.getDataType() == H5::PredType::IEEE_F32BE);

        // TODO Check whether shape is consistent
        const auto shape = ctx.shape(n);

        std::vector<float> buffer;

        using nncc::core::ADT::tensor::num_elements;
        buffer.resize(num_elements(shape));

        dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

        using nncc::core::ADT::tensor::Index;
        using nncc::core::ADT::tensor::IndexEnumerator;
        using nncc::core::ADT::tensor::LexicalLayout;

        LexicalLayout layout{};

        for (IndexEnumerator e{shape}; e.valid(); e.advance())
        {
          auto i = e.current();
          t.at(i) = buffer[layout.offset(shape, i)];
        }

        // TODO Check name
      };

      try
      {
        ctx.getMutableFloatTensor(n, fn);
      }
      catch (const H5::FileIException &)
      {
        // Skip if data is not present in HDF5 file
      }
    }
  }

private:
  H5::H5File _file;
  H5::Group _value_grp;
};

#include <nnkit/CmdlineArguments.h>

#include <memory>

extern "C" std::unique_ptr<nnkit::Action> make_action(const nnkit::CmdlineArguments &args)
{
  return std::make_unique<HD5ImportAction>(args.at(0));
}
