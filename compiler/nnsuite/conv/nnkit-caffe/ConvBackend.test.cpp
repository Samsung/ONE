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

#include "ConvBackend.h"

#include <nncc/core/ADT/kernel/Overlay.h>
#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <nncc/core/ADT/tensor/Overlay.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <morph/caffe.h>

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

class TestModel : public nnsuite::conv::Model
{
public:
  TestModel(const std::string &ifm_name, const feature::Shape &ifm_shape,
            const std::string &ofm_name, const feature::Shape &ofm_shape,
            const kernel::Shape &ker_shape, const kernel::Layout &ker_layout, float *ker_data)
    : _ifm_name(ifm_name), _ifm_shape(ifm_shape), _ofm_name(ofm_name), _ofm_shape(ofm_shape),
      _ker{ker_shape, ker_layout, ker_data}
  {
    // DO NOTHING
  }

public:
  const std::string &ifm_name(void) const override { return _ifm_name; }
  const feature::Shape &ifm_shape(void) const override { return _ifm_shape; }

public:
  const std::string &ofm_name(void) const override { return _ofm_name; }
  const feature::Shape &ofm_shape(void) const override { return _ofm_shape; }

public:
  const kernel::Shape &ker_shape(void) const override { return _ker.shape(); }
  const kernel::Reader<float> &ker_data(void) const override { return _ker; }

private:
  const std::string _ifm_name;
  const feature::Shape _ifm_shape;

private:
  const std::string _ofm_name;
  const feature::Shape _ofm_shape;

private:
  const kernel::Overlay<float, float *> _ker;
};

TEST(CONV_BACKEND, conv_3x3)
{
  const std::string ofm_name{"ofm"};
  const feature::Shape ofm_shape{1, 1, 1};
  float ofm_data[1] = {204.0f}; // EXPECTED

  const std::string ifm_name{"ifm"};
  const feature::Shape ifm_shape{1, 3, 3};
  float ifm_data[9] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  const kernel::Shape ker_shape{1, 1, 3, 3};
  float ker_data[9] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  using kernel::NCHWLayout;
  using tensor::LexicalLayout;

  TestModel model{ifm_name, ifm_shape, ofm_name, ofm_shape, ker_shape, NCHWLayout{}, ker_data};

  auto backend = ConvBackend::create(model);

  backend->prepare([&](nnkit::TensorContext &ctx) {
    ASSERT_EQ(ctx.size(), 1);
    ASSERT_EQ(ctx.name(0), ifm_name);
    // TODO Check IFM shape

    auto fill = [&](const nnkit::TensorContext &, uint32_t, tensor::Accessor<float> &t) {
      const auto tensor_shape = morph::caffe::as_tensor_shape(ifm_shape);
      const auto overlay = tensor::make_overlay<float, LexicalLayout>(tensor_shape, ifm_data);

      for (tensor::IndexEnumerator e{tensor_shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        t.at(index) = overlay.at(index);
      }
    };

    ctx.getMutableFloatTensor(0, fill);
  });

  backend->run();

  backend->teardown([&](nnkit::TensorContext &ctx) {
    ASSERT_EQ(ctx.size(), 1);
    ASSERT_EQ(ctx.name(0), ofm_name);

    auto verify = [&](const nnkit::TensorContext &, uint32_t, const tensor::Reader<float> &t) {
      const auto tensor_shape = morph::caffe::as_tensor_shape(ofm_shape);
      const auto overlay = tensor::make_overlay<float, LexicalLayout>(tensor_shape, ofm_data);

      for (tensor::IndexEnumerator e{tensor_shape}; e.valid(); e.advance())
      {
        const auto &index = e.current();
        EXPECT_EQ(t.at(index), overlay.at(index));
      }
    };

    ctx.getConstFloatTensor(0, verify);
  });
}
