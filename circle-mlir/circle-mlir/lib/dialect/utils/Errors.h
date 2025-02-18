/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/* various files and schema

#ifndef __CIRCLE_MLIR_DIALECT_UTILS_ERRORS_H__
#define __CIRCLE_MLIR_DIALECT_UTILS_ERRORS_H__

namespace mlir
{
namespace Circle
{

enum Code : int
{
  OK = 0,
  ERROR = 1,
};

class Status
{
public:
  Status() : _code{Code::OK} {};
  Status(Code code) : _code{code} {};

public:
  bool ok() const { return _code == Code::OK; }

private:
  Code _code = Code::OK;
};

inline Status OkStatus() { return Status(); }

#define CIR_PREDICT_FALSE(x) (x)

#define CIR_RETURN_IF_ERROR(...)          \
  do                                      \
  {                                       \
    Status _status = (__VA_ARGS__);       \
    if (CIR_PREDICT_FALSE(!_status.ok())) \
      return _status;                     \
  } while (0)

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_UTILS_ERRORS_H__
