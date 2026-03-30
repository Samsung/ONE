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

#include <caffe/proto/caffe.pb.h>

#include <nnkit/support/caffe/Backend.h>

#include <nncc/core/ADT/kernel/Overlay.h>
#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <memory>

using std::make_unique;

std::unique_ptr<nnkit::Backend> ConvBackend::create(const nnsuite::conv::Model &model)
{
  ::caffe::NetParameter param;

  param.set_name("conv");

  // Create 'Input' layer
  {
    auto input = param.add_layer();
    input->set_name("input");
    input->set_type("Input");
    input->add_top(model.ifm_name());

    auto input_param = new ::caffe::InputParameter{};
    auto input_shape = input_param->add_shape();
    input_shape->add_dim(1);
    input_shape->add_dim(model.ifm_shape().depth());
    input_shape->add_dim(model.ifm_shape().height());
    input_shape->add_dim(model.ifm_shape().width());
    input->set_allocated_input_param(input_param);
  }

  // Create 'Convolution' layer
  {
    auto conv = param.add_layer();
    conv->set_name("conv");
    conv->set_type("Convolution");
    conv->add_bottom(model.ifm_name());
    conv->add_top(model.ofm_name());

    const auto &ker_shape = model.ker_shape();

    auto ker_blob_shape = new ::caffe::BlobShape{};

    ker_blob_shape->add_dim(ker_shape.count());
    ker_blob_shape->add_dim(ker_shape.depth());
    ker_blob_shape->add_dim(ker_shape.height());
    ker_blob_shape->add_dim(ker_shape.width());

    auto ker_blob = conv->add_blobs();

    for (uint32_t n = 0; n < ker_shape.count(); ++n)
    {
      for (uint32_t ch = 0; ch < ker_shape.depth(); ++ch)
      {
        for (uint32_t row = 0; row < ker_shape.height(); ++row)
        {
          for (uint32_t col = 0; col < ker_shape.width(); ++col)
          {
            ker_blob->add_data(model.ker_data().at(n, ch, row, col));
          }
        }
      }
    }

    ker_blob->set_allocated_shape(ker_blob_shape);

    auto conv_param = new ::caffe::ConvolutionParameter{};
    conv_param->set_num_output(model.ker_shape().count());
    conv_param->set_bias_term(false);
    conv_param->add_kernel_size(model.ker_shape().height());
    conv_param->add_kernel_size(model.ker_shape().width());
    conv->set_allocated_convolution_param(conv_param);
  }

  auto net = make_unique<::caffe::Net<float>>(param);
  return make_unique<nnkit::support::caffe::Backend<float>>(std::move(net));
}
