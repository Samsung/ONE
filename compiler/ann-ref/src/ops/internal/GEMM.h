/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __GEMM_H__
#define __GEMM_H__

#include "Eigen/Core"

template <typename Lhs, typename Rhs, typename Result>
void Gemm(const Eigen::MatrixBase<Lhs> &lhs, const Eigen::MatrixBase<Rhs> &rhs,
          Eigen::MatrixBase<Result> *result)
{
  if (rhs.cols() == 1)
  {
    result->col(0).noalias() = lhs * rhs.col(0);
  }
  else
  {
    result->noalias() = lhs * rhs;
  }
}


#endif // __GEMM_H__
