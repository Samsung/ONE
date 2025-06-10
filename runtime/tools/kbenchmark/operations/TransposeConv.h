/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __KBENCHMARK_OPERATIONS_TRANSPOSE_CONV_H__
#define __KBENCHMARK_OPERATIONS_TRANSPOSE_CONV_H__

#include "Operation.h"
#include "Utils.h"

namespace kbenchmark
{
namespace operation
{

class TransposeConv final : public Operation
{
public:
  TransposeConv() = default;

  nonius::parameters params(int layer_num, OperationInfo &info) override
  {
    nonius::parameters params;

    params.insert({"LAYER", nonius::param{layer_num}});

    params.insert({"BATCH", nonius::param{1}});

    // TODO The output_shape will be used in Dynamic tensor case
    auto _output_shape = get_key_dims({"input0"}, info);

    auto _weights = get_key_dims({"input1"}, info);
    params.insert({"KER_H", nonius::param{_weights[1]}});
    params.insert({"KER_W", nonius::param{_weights[2]}});

    auto _input = get_key_dims({"input2"}, info);
    params.insert({"IFM_H", nonius::param{_input[1]}});
    params.insert({"IFM_W", nonius::param{_input[2]}});
    params.insert({"IFM_C", nonius::param{_input[3]}});

    auto _output = get_key_dims({"output0"}, info);
    params.insert({"OFM_H", nonius::param{_output[1]}});
    params.insert({"OFM_W", nonius::param{_output[2]}});
    params.insert({"OFM_C", nonius::param{_output[3]}});

    auto _stride_h = get_key_int({"stride_h"}, info);
    auto _stride_w = get_key_int({"stride_w"}, info);
    params.insert({"STRIDE_H", nonius::param{_stride_h}});
    params.insert({"STRIDE_W", nonius::param{_stride_w}});

    auto _pad = get_key_string({"padding"}, info);
    params.insert({"PADDING", nonius::param{_pad}});

    return params;
  }
};

} // namespace operation
} // namespace kbenchmark

#endif // __KBENCHMARK_OPERATIONS_TRANSPOSE_CONV_H__
