/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cassert>
#include <string>
#include <iostream>

#include "nnfw.h"
#include "nnfw_experimental.h"

namespace onert_train
{
uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    assert(ti->dims[i] >= 0);
    n *= ti->dims[i];
  }
  return n;
}

uint64_t bufsize_for(const nnfw_tensorinfo *ti)
{
  static int elmsize[] = {
    sizeof(float),   /* NNFW_TYPE_TENSOR_FLOAT32 */
    sizeof(int),     /* NNFW_TYPE_TENSOR_INT32 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_QUANT8_ASYMM */
    sizeof(bool),    /* NNFW_TYPE_TENSOR_BOOL = 3 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_UINT8 = 4 */
    sizeof(int64_t), /* NNFW_TYPE_TENSOR_INT64 = 5 */
    sizeof(int8_t),  /* NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED = 6 */
    sizeof(int16_t), /* NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED = 7 */
  };
  return elmsize[ti->dtype] * num_elems(ti);
}

std::ostream &operator<<(std::ostream &os, const NNFW_TRAIN_OPTIMIZER &opt)
{
  switch (opt)
  {
    case NNFW_TRAIN_OPTIMIZER_ADAM:
      os << "adam";
      break;
    case NNFW_TRAIN_OPTIMIZER_SGD:
      os << "sgd";
      break;
    default:
      os << "unsupported optimizer";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const NNFW_TRAIN_LOSS &loss)
{
  switch (loss)
  {
    case NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR:
      os << "mean squared error";
      break;
    case NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY:
      os << "categorical crossentropy";
      break;
    default:
      os << "unsupported loss";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const NNFW_TRAIN_LOSS_REDUCTION &loss_reduction)
{
  switch (loss_reduction)
  {
    case NNFW_TRAIN_LOSS_REDUCTION_INVALID:
      os << "use default setting";
      break;
    case NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE:
      os << "sum over batch size";
      break;
    case NNFW_TRAIN_LOSS_REDUCTION_SUM:
      os << "sum";
      break;
    default:
      os << "unsupported reduction type";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const nnfw_loss_info &loss_info)
{
  os << "{loss = " << loss_info.loss << ", reduction = " << loss_info.reduction_type << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const nnfw_train_info &info)
{
  os << "- learning_rate   = " << info.learning_rate << "\n";
  os << "- batch_size      = " << info.batch_size << "\n";
  os << "- loss_info       = " << info.loss_info << "\n";
  os << "- optimizer       = " << info.opt << "\n";
  return os;
}

} // namespace onert_train
