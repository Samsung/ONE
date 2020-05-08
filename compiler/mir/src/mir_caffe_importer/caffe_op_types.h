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

#ifndef MIR_CAFFE_OP_TYPES_H
#define MIR_CAFFE_OP_TYPES_H

namespace mir_caffe
{

enum class CaffeOpType
{
  absVal,
  accuracy,
  argMax,
  batchNorm,
  batchReindex,
  bias,
  BNLL,
  clip,
  concat,
  contrastiveLoss,
  convolution,
  crop,
  data,
  deconvolution,
  dropout,
  dummyData,
  eltwise,
  ELU,
  embed,
  euclidianLoss,
  exp,
  filter,
  flatten,
  HDF5Data,
  HDF5Output,
  hingeLoss,
  im2Col,
  imageData,
  infogainLoss,
  innerProduct,
  input,
  log,
  LRN,
  LSTM,
  memoryData,
  multinomialLogisticLoss,
  MVN,
  parameter,
  pooling,
  power,
  PReLU,
  python,
  recurrent,
  reduction,
  ReLU,
  reshape,
  RNN,
  scale,
  sigmoidCrossEntropyLoss,
  sigmoid,
  silence,
  slice,
  softmax,
  softmaxWithLoss,
  split,
  SPP,
  tanh,
  threshold,
  tile,
  windowData
};

} // namespace mir_caffe

#endif // MIR_CAFFE_OP_TYPES_H
