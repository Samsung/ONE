/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_EGIEN_EIGEN_SPATIAL_CONVOLUTIONS_H__
#define __NNFW_CKER_EGIEN_EIGEN_SPATIAL_CONVOLUTIONS_H__

// #define EIGEN_USE_CUSTOM_THREAD_POOL
#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

// Note the following header is used in both TF and TFLite. Particularly, it's
// used for float TFLite Conv2D.
#include "cker/eigen/eigen_spatial_convolutions-inl.h"

#endif // __NNFW_CKER_EGIEN_EIGEN_SPATIAL_CONVOLUTIONS_H__
