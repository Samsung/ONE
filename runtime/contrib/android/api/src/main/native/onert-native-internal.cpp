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

#include "onert-native-internal.h"

namespace
{

std::unordered_map<nnfw_session *, jni::TempOutputMap> g_sess_2_output;

inline nnfw_session *getSession(Handle handle) { return reinterpret_cast<nnfw_session *>(handle); }

inline jni::SessionMap &getSessionMap() { return g_sess_2_output; }

inline bool containsInTempOutput(nnfw_session *sess)
{
  return (g_sess_2_output.find(sess) != g_sess_2_output.end());
}

inline bool containsInTempOutput(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  return containsInTempOutput(sess);
}

inline jni::TempOutputMap &getTempOutputMap(nnfw_session *sess) { return g_sess_2_output.at(sess); }

inline jni::TempOutputMap &getTempOutputMap(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  return getTempOutputMap(sess);
}

size_t getByteSizeOfDataType(NNFW_TYPE dtype)
{
  size_t size = 0;
  switch (dtype)
  {
    case NNFW_TYPE_TENSOR_FLOAT32:
    case NNFW_TYPE_TENSOR_INT32:
      size = 4;
      break;
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
    case NNFW_TYPE_TENSOR_BOOL:
    case NNFW_TYPE_TENSOR_UINT8:
      size = 1;
      break;
    default:
      break;
  }
  return size;
}

size_t getByteSize(const nnfw_tensorinfo &tensor_info)
{
  size_t size = getByteSizeOfDataType(tensor_info.dtype);
  for (auto i = 0; i < tensor_info.rank; ++i)
  {
    size *= tensor_info.dims[i];
  }
  return size;
}

} // namespace

namespace jni
{

Handle createSession()
{
  nnfw_session *sess = nullptr;
  if (nnfw_create_session(&sess) == NNFW_STATUS_ERROR)
  {
    return 0; // nullptr
  }
  return reinterpret_cast<Handle>(sess);
}

void closeSession(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  nnfw_close_session(sess);
}

bool loadModel(Handle handle, const char *nnpkg_path)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_load_model_from_file(sess, nnpkg_path) == NNFW_STATUS_NO_ERROR);
}

bool prepare(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_prepare(sess) == NNFW_STATUS_NO_ERROR);
}

bool run(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_run(sess) == NNFW_STATUS_NO_ERROR);
}

bool setInput(Handle handle, const TensorParams &params)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_set_input(sess, params.index, params.type, params.buffer, params.buffer_size) ==
          NNFW_STATUS_NO_ERROR);
}

bool setOutput(Handle handle, TensorParams &params)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_set_output(sess, params.index, params.type, params.buffer, params.buffer_size) ==
          NNFW_STATUS_NO_ERROR);
}

bool setInputLayout(Handle handle, const LayoutParams &params)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_set_input_layout(sess, params.index, params.layout) == NNFW_STATUS_NO_ERROR);
}

bool setOutputLayout(Handle handle, const LayoutParams &params)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_set_output_layout(sess, params.index, params.layout) == NNFW_STATUS_NO_ERROR);
}

int getInputSize(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  uint32_t size = 0;
  if (nnfw_input_size(sess, &size) == NNFW_STATUS_NO_ERROR)
    return static_cast<int>(size);
  else
    return -1;
}

int getOutputSize(Handle handle)
{
  nnfw_session *sess = getSession(handle);
  uint32_t size = 0;
  if (nnfw_output_size(sess, &size) == NNFW_STATUS_NO_ERROR)
    return static_cast<int>(size);
  else
    return -1;
}

bool setAvailableBackends(Handle handle, const char *backends)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_set_available_backends(sess, backends) == NNFW_STATUS_NO_ERROR);
}

bool getInputTensorInfo(Handle handle, uint32_t index, TensorInfo &info)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_input_tensorinfo(sess, index, &info) == NNFW_STATUS_NO_ERROR);
}

bool getOutputTensorInfo(Handle handle, uint32_t index, TensorInfo &info)
{
  nnfw_session *sess = getSession(handle);
  return (nnfw_output_tensorinfo(sess, index, &info) == NNFW_STATUS_NO_ERROR);
}

bool newTempOutputBuf(Handle handle, uint32_t index)
{
  nnfw_session *sess = getSession(handle);

  TensorInfo tensor_info;
  if (nnfw_output_tensorinfo(sess, index, &tensor_info) == NNFW_STATUS_ERROR)
    return false;

  auto bufsize = getByteSize(tensor_info);

  if (containsInTempOutput(sess) == false)
  {
    TempOutputMap tom;
    tom.emplace(index, TempOutput{new char[bufsize]{}, bufsize, tensor_info.dtype});
    getSessionMap().emplace(sess, tom);
  }
  else
  {
    auto &tom = getSessionMap().at(sess);
    tom.emplace(index, TempOutput{new char[bufsize]{}, bufsize, tensor_info.dtype});
  }

  return true;
}

bool deleteTempOutputBuf(Handle handle, uint32_t index)
{
  nnfw_session *sess = getSession(handle);

  if (containsInTempOutput(sess) == false)
    return false;

  auto &tom = getSessionMap().at(sess);
  if (tom.find(index) == tom.end())
    return false;

  delete[] tom.at(index).buf;
  tom.erase(index);

  return true;
}

const TempOutput *getTempOutputBuf(Handle handle, uint32_t index)
{
  nnfw_session *sess = getSession(handle);

  if (containsInTempOutput(sess) == false)
    return nullptr;

  auto &tom = getSessionMap().at(sess);
  if (tom.find(index) == tom.end())
    return nullptr;

  return &(tom.at(index));
}

} // namespace jni
