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

#include "onert-native-api.h"

#include <android/log.h>

#include "onert-native-internal.h"
#include "onert-native-helper.h"

namespace
{

// android log tag
const char *JTAG = "ONERT_NATIVE";

} // namespace

JNIEXPORT jlong JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeCreateSession(JNIEnv *,
                                                                                        jobject)
{
  Handle sess = jni::createSession();
  if (sess == 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] nnfw_create_session is failed",
                        __PRETTY_FUNCTION__);
  }
  return sess;
}

JNIEXPORT void JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeCloseSession(JNIEnv *,
                                                                                      jobject,
                                                                                      jlong handle)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return;

  jni::closeSession(handle);
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeLoadModelFromFile(
  JNIEnv *env, jobject, jlong handle, jstring jnnpkg_path)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  const char *nnpkg_path = env->GetStringUTFChars(jnnpkg_path, 0);
  __android_log_print(ANDROID_LOG_DEBUG, JTAG, "%s] nnpkg_path: %s", __PRETTY_FUNCTION__,
                      nnpkg_path);

  bool result = jni::loadModel(handle, nnpkg_path);

  env->ReleaseStringUTFChars(jnnpkg_path, nnpkg_path);

  if (result == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativePrepare(JNIEnv *,
                                                                                     jobject,
                                                                                     jlong handle)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  if (jni::prepare(handle) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeRun(JNIEnv *, jobject,
                                                                                 jlong handle)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  if (jni::run(handle) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetInput(
  JNIEnv *env, jobject, jlong handle, jint jindex, jint jtype, jobject jbuf, jint jbufsize)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  jni::TensorParams params;
  if (jni_helper::getTensorParams(env, jindex, jtype, jbuf, jbufsize, params) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed getTensorParams", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] index(%d), type(%d), buf(%p), buf_sz(%lu)",
                      __PRETTY_FUNCTION__, params.index, params.type, params.buffer,
                      params.buffer_size);

  if (jni::setInput(handle, params) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed native setInput", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetOutput(
  JNIEnv *env, jobject, jlong handle, jint jindex, jint jtype, jobject jbuf, jint jbufsize)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  jni::TensorParams params;
  if (jni_helper::getTensorParams(env, jindex, jtype, jbuf, jbufsize, params) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed getTensorParams", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] index(%d), type(%d), buf(%p), buf_sz(%lu)",
                      __PRETTY_FUNCTION__, params.index, params.type, params.buffer,
                      params.buffer_size);

  if (jni::setOutput(handle, params) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed native setOutput",
                        __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetInputLayout(
  JNIEnv *, jobject, jlong handle, jint jindex, jint jlayout)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  jni::LayoutParams params;
  if (jni_helper::getLayoutParams(jindex, jlayout, params) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  if (jni::setInputLayout(handle, params) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetOutputLayout(
  JNIEnv *, jobject, jlong handle, jint jindex, jint jlayout)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  jni::LayoutParams params;
  if (jni_helper::getLayoutParams(jindex, jlayout, params) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  if (jni::setOutputLayout(handle, params) == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

JNIEXPORT jint JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetInputSize(JNIEnv *,
                                                                                      jobject,
                                                                                      jlong handle)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return -1;

  int size = 0;
  if ((size = jni::getInputSize(handle)) < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return -1;
  }

  return static_cast<jint>(size);
}

JNIEXPORT jint JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetOutputSize(JNIEnv *,
                                                                                       jobject,
                                                                                       jlong handle)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return -1;

  int size = 0;
  if ((size = jni::getOutputSize(handle)) < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return -1;
  }

  return static_cast<jint>(size);
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetAvailableBackends(
  JNIEnv *env, jobject, jlong handle, jstring jbackends)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  const char *backends = env->GetStringUTFChars(jbackends, 0);
  __android_log_print(ANDROID_LOG_DEBUG, JTAG, "%s] backends: %s", __PRETTY_FUNCTION__, backends);

  auto result = jni::setAvailableBackends(handle, backends);

  env->ReleaseStringUTFChars(jbackends, backends);

  if (result == false)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetInputTensorInfo(
  JNIEnv *env, jobject, jlong handle, jint jindex, jobject jinfo)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  jni::TensorInfo tensor_info;
  if (jni_helper::getInputTensorInfo(handle, jindex, tensor_info) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  if (jni_helper::setTensorInfoToJava(env, tensor_info, jinfo) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetOutputTensorInfo(
  JNIEnv *env, jobject, jlong handle, jint jindex, jobject jinfo)
{
  if (jni_helper::verifyHandle(handle) == JNI_FALSE)
    return JNI_FALSE;

  jni::TensorInfo tensor_info;
  if (jni_helper::getOutputTensorInfo(handle, jindex, tensor_info) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  if (jni_helper::setTensorInfoToJava(env, tensor_info, jinfo) == JNI_FALSE)
  {
    __android_log_print(ANDROID_LOG_ERROR, JTAG, "%s] failed", __PRETTY_FUNCTION__);
    return JNI_FALSE;
  }

  return JNI_TRUE;
}
