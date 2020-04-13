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

#ifndef HANDLE_OP
#error "You should define HANDLE_OP before including this file"
#endif // HANDLE_OP

HANDLE_OP(abs, AbsOp)
HANDLE_OP(add, AddOp)
HANDLE_OP(avgPool2D, AvgPool2DOp)
HANDLE_OP(broadcast, BroadcastOp)
HANDLE_OP(cappedReLU, CappedReluOp)
HANDLE_OP(concat, ConcatOp)
HANDLE_OP(constant, ConstantOp)
HANDLE_OP(conv2D, Conv2DOp)
HANDLE_OP(deConv2D, DeConv2DOp)
HANDLE_OP(depthwiseConv, DepthwiseConv2DOp)
HANDLE_OP(dequantize, DequantizeOp)
HANDLE_OP(div, DivOp)
HANDLE_OP(ELU, EluOp)
HANDLE_OP(equal, EqualOp)
HANDLE_OP(fullyConnected, FullyConnectedOp)
HANDLE_OP(gather, GatherOp)
HANDLE_OP(greater, GreaterOp)
HANDLE_OP(hardswish, HardSwishOp)
HANDLE_OP(input, InputOp)
HANDLE_OP(leakyReLU, LeakyReluOp)
HANDLE_OP(less, LessOp)
HANDLE_OP(max, MaxOp)
HANDLE_OP(maxPool2D, MaxPool2DOp)
HANDLE_OP(mul, MulOp)
HANDLE_OP(output, OutputOp)
HANDLE_OP(pad, PadOp)
HANDLE_OP(quantize, QuantizeOp)
HANDLE_OP(reduceMean, ReduceMeanOp)
HANDLE_OP(ReLU, ReluOp)
HANDLE_OP(reshape, ReshapeOp)
HANDLE_OP(resizeIm, ResizeOp)
HANDLE_OP(sigmoid, SigmoidOp)
HANDLE_OP(slice, SliceOp)
HANDLE_OP(softmax, SoftmaxOp)
HANDLE_OP(sqrt, SqrtOp)
HANDLE_OP(squeeze, SqueezeOp)
HANDLE_OP(sub, SubOp)
HANDLE_OP(tanh, TanhOp)
HANDLE_OP(transpose, TransposeOp)
