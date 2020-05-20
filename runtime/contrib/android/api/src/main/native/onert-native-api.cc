#include "onert-native-api.h"

#include <android/log.h>
#include <nnfw.h>

static const char *TAG = "ONERT_NATIVE";

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeCreateSession
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeCreateSession
  (JNIEnv *env, jobject thiz)
{
  nnfw_session *sess = nullptr;
  if (nnfw_create_session(&sess) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_create_session is failed", __PRETTY_FUNCTION__);
    return -1;
  }
  jlong ret = reinterpret_cast<jlong>(sess);
  return ret;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeCloseSession
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeCloseSession
  (JNIEnv *env, jobject thiz, jlong handle)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return;
  }
  nnfw_close_session(sess);
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeLoadModelFromFile
 * Signature: (JLjava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeLoadModelFromFile
  (JNIEnv *env, jobject thiz, jlong handle, jstring jnnpkg_path)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  const char *nnpkg_path = env->GetStringUTFChars(jnnpkg_path, 0);
  __android_log_print(ANDROID_LOG_DEBUG, TAG, "%s] nnpkg_path: %s", __PRETTY_FUNCTION__, nnpkg_path);

  NNFW_STATUS result = nnfw_load_model_from_file(sess, nnpkg_path);
  env->ReleaseStringUTFChars(jnnpkg_path, nnpkg_path);
  if (result == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_load_model_from_file is failed", __PRETTY_FUNCTION__);
    return false;
  }
  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativePrepare
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativePrepare
  (JNIEnv *env, jobject thiz, jlong handle)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (nnfw_prepare(sess) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_prepare is failed", __PRETTY_FUNCTION__);
    return false;
  }
  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeRun
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeRun
  (JNIEnv *env, jobject thiz, jlong handle)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (nnfw_run(sess) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_run is failed", __PRETTY_FUNCTION__);
    return false;
  }
  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeSetInput
 * Signature: (JIILjava/nio/ByteBuffer;I)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetInput
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jint jtype, jobject jbuf, jint jbufsize)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__, jindex);
    return false;
  }
  uint32_t index = static_cast<uint32_t>(jindex);

  if (jtype < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] type(%d) is wrong", __PRETTY_FUNCTION__, jtype);
    return false;
  }
  NNFW_TYPE type = static_cast<NNFW_TYPE>(jtype);

  jbyte *buffer = (jbyte *) env->GetDirectBufferAddress(jbuf);
  if (buffer == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] buffer is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (jbufsize < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] length(%d) is wrong", __PRETTY_FUNCTION__, jbufsize);
    return false;
  }
  size_t length = static_cast<size_t>(jbufsize);

  if (nnfw_set_input(sess, index, type, buffer, length) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] nnfw_set_input is failed", __PRETTY_FUNCTION__);
    return false;

  }

  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeSetOutput
 * Signature: (JIILjava/nio/ByteBuffer;I)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetOutput
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jint jtype, jobject jbuf, jint jbufsize)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__, jindex);
    return false;
  }
  uint32_t index = static_cast<uint32_t>(jindex);

  if (jtype < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] type(%d) is wrong", __PRETTY_FUNCTION__, jtype);
    return false;
  }
  NNFW_TYPE type = static_cast<NNFW_TYPE>(jtype);

  jbyte *buffer = (jbyte *) env->GetDirectBufferAddress(jbuf);
  if (buffer == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] buffer is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (jbufsize < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] length(%d) is wrong", __PRETTY_FUNCTION__, jbufsize);
    return false;
  }
  size_t length = static_cast<size_t>(jbufsize);

  if (nnfw_set_output(sess, index, type, buffer, length) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] nnfw_set_output is failed", __PRETTY_FUNCTION__);
    return false;

  }

  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeSetInputLayout
 * Signature: (JII)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetInputLayout
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jint jlayout)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__, jindex);
    return false;
  }
  uint32_t index = static_cast<uint32_t>(jindex);

  if (jlayout < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] layout(%d) is wrong", __PRETTY_FUNCTION__, jlayout);
    return false;
  }
  NNFW_LAYOUT layout = static_cast<NNFW_LAYOUT>(jlayout);

  if (nnfw_set_input_layout(sess, index, layout) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] nnfw_set_input_layout is failed", __PRETTY_FUNCTION__);
    return false;
  }

  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeSetOutputLayout
 * Signature: (JII)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetOutputLayout
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jint jlayout)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__, jindex);
    return false;
  }
  uint32_t index = static_cast<uint32_t>(jindex);

  if (jlayout < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] layout(%d) is wrong", __PRETTY_FUNCTION__, jlayout);
    return false;
  }
  NNFW_LAYOUT layout = static_cast<NNFW_LAYOUT>(jlayout);

  if (nnfw_set_output_layout(sess, index, layout) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] nnfw_set_output_layout is failed", __PRETTY_FUNCTION__);
    return false;
  }

  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeGetInputSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetInputSize
  (JNIEnv *env, jobject thiz, jlong handle)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  uint32_t number = 0;
  if (nnfw_input_size(sess, &number) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] nnfw_input_size is failed", __PRETTY_FUNCTION__);
    return -1;
  }

  return static_cast<jint>(number);
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeGetOutputSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetOutputSize
  (JNIEnv *env, jobject thiz, jlong handle)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  uint32_t number = 0;
  if (nnfw_output_size(sess, &number) == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] nnfw_output_size is failed", __PRETTY_FUNCTION__);
    return -1;
  }

  return static_cast<jint>(number);
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeSetAvailableBackends
 * Signature: (JLjava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetAvailableBackends
  (JNIEnv *env, jobject thiz, jlong handle, jstring jbackends)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return false;
  }

  const char *backends = env->GetStringUTFChars(jbackends, 0);
  __android_log_print(ANDROID_LOG_DEBUG, TAG, "%s] backends: %s", __PRETTY_FUNCTION__, backends);

  NNFW_STATUS result = nnfw_set_available_backends(sess, backends);
  env->ReleaseStringUTFChars(jbackends, backends);
  if (result == NNFW_STATUS_ERROR)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_set_available_backends is failed", __PRETTY_FUNCTION__);
    return false;
  }
  return true;
}
