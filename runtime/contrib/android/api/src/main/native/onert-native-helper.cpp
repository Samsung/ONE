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

#include "onert-native-helper.h"

#include <android/log.h>

namespace
{

// android log tag
const char *TAG = "ONERT_NATIVE";

bool g_reflect_cached = false;
struct ReflectInfo
{
  jclass info;
  jfieldID type;
  jfieldID rank;
  jfieldID shape;
};
ReflectInfo g_cached_reflect_info;

jboolean cacheReflectInfo(JNIEnv *env)
{
  jclass info_cls = env->FindClass("com/samsung/onert/NativeSessionWrapper$InternalTensorInfo");
  if (info_cls == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] java info class is failed",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  jfieldID type_fld = env->GetFieldID(info_cls, "type", "I");
  if (type_fld == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] TensorInfo's type field id is failed",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  jfieldID rank_fld = env->GetFieldID(info_cls, "rank", "I");
  if (rank_fld == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] TensorInfo's rank field id is failed",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  jfieldID shape_fld = env->GetFieldID(info_cls, "shape", "[I");
  if (shape_fld == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] TensorInfo's shape field id is failed",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  g_cached_reflect_info.info = info_cls;
  g_cached_reflect_info.type = type_fld;
  g_cached_reflect_info.rank = rank_fld;
  g_cached_reflect_info.shape = shape_fld;
  g_reflect_cached = true;
  return JNI_TRUE;
}

} // namespace

namespace jni_helper
{

jboolean verifyHandle(jlong handle)
{
  if (handle == 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] handle is null", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

jboolean getTensorParams(JNIEnv *env, jint jindex, jint jtype, jobject jbuf, jint jbufsize,
                         jni::TensorParams &params)
{
  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__,
                        jindex);
    return JNI_FALSE;
  }
  params.index = static_cast<uint32_t>(jindex);

  if (jtype < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] type(%d) is wrong", __PRETTY_FUNCTION__,
                        jtype);
    return JNI_FALSE;
  }
  params.type = static_cast<NNFW_TYPE>(jtype);

  jbyte *buffer = reinterpret_cast<jbyte *>(env->GetDirectBufferAddress(jbuf));
  if (buffer == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] buffer is null", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  params.buffer = buffer;

  if (jbufsize < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] length(%d) is wrong", __PRETTY_FUNCTION__,
                        jbufsize);
    return JNI_FALSE;
  }
  params.buffer_size = static_cast<size_t>(jbufsize);

  return JNI_TRUE;
}

jboolean getTensorParams(jint jindex, jint jtype, jlong handle, jni::TensorParams &params)
{
  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__,
                        jindex);
    return JNI_FALSE;
  }
  auto index = static_cast<uint32_t>(jindex);

  if (jtype < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] type(%d) is wrong", __PRETTY_FUNCTION__,
                        jtype);
    return JNI_FALSE;
  }
  auto type = static_cast<NNFW_TYPE>(jtype);

  const jni::TempOutput *to = jni::getTempOutputBuf(handle, index);
  if (to == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] failed to get TempOutput",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  params.index = index;
  params.type = type;
  params.buffer = to->buf;
  params.buffer_size = to->bufsize;

  return JNI_TRUE;
}

jboolean getLayoutParams(jint jindex, jint jlayout, jni::LayoutParams &params)
{
  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__,
                        jindex);
    return JNI_FALSE;
  }
  params.index = static_cast<uint32_t>(jindex);

  if (jlayout < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] layout(%d) is wrong", __PRETTY_FUNCTION__,
                        jlayout);
    return JNI_FALSE;
  }
  params.layout = static_cast<NNFW_LAYOUT>(jlayout);

  return JNI_TRUE;
}

jboolean setTensorInfoToJava(JNIEnv *env, const nnfw_tensorinfo &tensor_info, jobject jinfo)
{
  if (g_reflect_cached == false)
  {
    if (cacheReflectInfo(env) == JNI_FALSE)
    {
      __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] failed", __PRETTY_FUNCTION__);
      return JNI_FALSE;
    }
  }

  jclass info_cls = g_cached_reflect_info.info;

  jfieldID type_fld = g_cached_reflect_info.type;
  jint jtype = static_cast<jint>(tensor_info.dtype);
  env->SetIntField(jinfo, type_fld, jtype);

  jfieldID rank_fld = g_cached_reflect_info.rank;
  jint jrank = tensor_info.rank;
  env->SetIntField(jinfo, rank_fld, jrank);

  jfieldID shape_fld = g_cached_reflect_info.shape;
  jintArray jshape = env->NewIntArray(jrank);
  if (jshape == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] TensorInfo's shape[] allocation is failed",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  env->SetIntArrayRegion(jshape, 0, jrank, (tensor_info.dims));
  env->SetObjectField(jinfo, shape_fld, jshape);

  return JNI_TRUE;
}

jboolean getInputTensorInfo(jlong handle, jint jindex, jni::TensorInfo &info)
{
  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__,
                        jindex);
    return JNI_FALSE;
  }
  auto index = static_cast<uint32_t>(jindex);

  if (jni::getInputTensorInfo(handle, index, info) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

jboolean getOutputTensorInfo(jlong handle, jint jindex, jni::TensorInfo &info)
{
  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__,
                        jindex);
    return JNI_FALSE;
  }
  auto index = static_cast<uint32_t>(jindex);

  if (jni::getOutputTensorInfo(handle, index, info) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

} // namespace jni_helper
