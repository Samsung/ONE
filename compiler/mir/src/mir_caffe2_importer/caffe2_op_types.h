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

#ifndef MIR_CAFFE2_OP_TYPES_H
#define MIR_CAFFE2_OP_TYPES_H

namespace mir_caffe2
{

enum class SupportedCaffe2OpType
{
  add,
  averagePool,
  clip,
  concat,
  conv,
  constantFill,
  dropout,
  FC,
  givenTensorFill,
  givenTensorInt64Fill,
  maxPool,
  mul,
  relu,
  reshape,
  resizeNearest,
  sigmoid,
  softmax,
  spatialBN,
  sum,
};

} // namespace mir_caffe2

#endif // MIR_CAFFE2_OP_TYPES_H
