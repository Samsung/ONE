/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLTypeInference.h"

#include "schema_generated.h"

#include "Dialect/Service/TFLTypeInferenceRule.h"
#include "Dialect/IR/TFLDialect.h"

#include <loco/IR/CanonicalNode.h>
#include <loco/IR/CanonicalNodeVisitor.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <locoex/COpDialect.h>
#include <locoex/Service/COpTypeInference.h>

#include <oops/InternalExn.h>

#include <stdexcept>
#include <type_traits>

namespace
{

tflite::TensorType translateLocoTypeToTFLite(loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::U8:
      return tflite::TensorType_UINT8;
    //  case loco::DataType::U16: unsupported
    //  case loco::DataType::U32: unsupported
    //  case loco::DataType::U64: unsupported
    case loco::DataType::S8:
      return tflite::TensorType_INT8;
    case loco::DataType::S16:
      return tflite::TensorType_INT16;
    case loco::DataType::S32:
      return tflite::TensorType_INT32;
    case loco::DataType::S64:
      return tflite::TensorType_INT64;
    case loco::DataType::FLOAT16:
      return tflite::TensorType_FLOAT16;
    case loco::DataType::FLOAT32:
      return tflite::TensorType_FLOAT32;
    //  case loco::DataType::FLOAT64: unsupported
    default:
      break;
  }

  INTERNAL_EXN_V("Trying to converte unsupported loco dtype", oops::to_uint32(dtype));
}

} // namespace

namespace exo
{

tflite::TensorType TypeInference::get(loco::Node *node)
{
  assert(loco::dtype_known(node));
  return translateLocoTypeToTFLite(loco::dtype_get(node));
}

} // namespace exo
