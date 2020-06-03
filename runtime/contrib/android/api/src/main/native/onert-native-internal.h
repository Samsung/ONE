/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

// Independent from jni
#pragma once

// onert
#include <nnfw.h> // TODO change nnfw.h to onert.h

// stl
#include <unordered_map>
#include <cassert>

// from jni_md.h
#ifdef _LP64 /* 64-bit Solaris */
typedef long Handle;
#else
typedef long long Handle;
#endif

namespace jni
{

// for output temp tensor
struct TempOutput
{
  char *buf;
  size_t bufsize;
  NNFW_TYPE type;
};
using TempOutputMap = std::unordered_map<uint32_t, TempOutput>;
using SessionMap = std::unordered_map<nnfw_session *, TempOutputMap>;

struct TensorParams
{
  uint32_t index;
  NNFW_TYPE type;
  void *buffer;
  size_t buffer_size;
};

struct LayoutParams
{
  uint32_t index;
  NNFW_LAYOUT layout;
};

using TensorInfo = nnfw_tensorinfo;

Handle createSession();
void closeSession(Handle handle);
bool loadModel(Handle handle, const char *nnpkg_path);
bool prepare(Handle handle);
bool run(Handle handle);
bool setInput(Handle handle, const TensorParams &params);
bool setOutput(Handle handle, TensorParams &params);
bool setInputLayout(Handle handle, const LayoutParams &params);
bool setOutputLayout(Handle handle, const LayoutParams &params);
int getInputSize(Handle handle);
int getOutputSize(Handle handle);
bool setAvailableBackends(Handle handle, const char *backends);
bool getInputTensorInfo(Handle handle, uint32_t index, TensorInfo &info);
bool getOutputTensorInfo(Handle handle, uint32_t index, TensorInfo &info);
bool newTempOutputBuf(Handle handle, uint32_t index);
bool deleteTempOutputBuf(Handle handle, uint32_t index);
const TempOutput *getTempOutputBuf(Handle handle, uint32_t index);

} // namespace jni
