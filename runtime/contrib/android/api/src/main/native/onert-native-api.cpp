#include "onert-native-api.h"

#include <android/log.h>
#include <nnfw.h>
#include <unordered_map>
#include <sstream>

namespace {

static const char *TAG = "ONERT_NATIVE";

// Session* -> vector<Output>
// Output: buf, bufsize, type
struct TempOutput {
  char *buf;
  size_t bufsize;
  NNFW_TYPE type;
};

using TempOutputMap = std::unordered_map<uint32_t, TempOutput>;
static std::unordered_map<nnfw_session *, TempOutputMap> g_sess_2_output;

} // namespace

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
  auto ret = reinterpret_cast<jlong>(sess);
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

  // TODO DEBUG
  __android_log_print(ANDROID_LOG_ERROR, TAG,
                      "%s] after nnfw_run output temp buffer", __PRETTY_FUNCTION__);

  auto &tom = g_sess_2_output.at(sess);
  for (const auto &it : tom) {
    std::stringstream ss;
    auto index = it.first;
    auto &output = it.second;
    ss << "Output #" << index << ": ";
    if (output.type == 0)
    {
      for (int i = 0; i < output.bufsize; i += 4) {
        float *fp = reinterpret_cast<float *>(&(output.buf[i]));
        ss << *fp << " ";
      }
    }
    else
    {
      for (int i = 0; i < output.bufsize; i += 4) {
        int *ip = reinterpret_cast<int *>(&(output.buf[i]));
        ss << *ip << " ";
      }
    }
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", ss.str().c_str());
  }
  __android_log_print(ANDROID_LOG_ERROR, TAG, "\n");

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
  auto index = static_cast<uint32_t>(jindex);

  if (jtype < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] type(%d) is wrong", __PRETTY_FUNCTION__, jtype);
    return false;
  }
  NNFW_TYPE type = static_cast<NNFW_TYPE>(jtype);

  jbyte *buffer = reinterpret_cast<jbyte *>(env->GetDirectBufferAddress(jbuf));
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
  auto length = static_cast<size_t>(jbufsize);

  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d), type(%d), buf(%p), buf_sz(%lu)",
                      __PRETTY_FUNCTION__, index, jtype, buffer, length);

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
 * Signature: (JII)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeSetOutput
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jint jtype)
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
  auto index = static_cast<uint32_t>(jindex);

  if (jtype < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] type(%d) is wrong", __PRETTY_FUNCTION__, jtype);
    return false;
  }
  NNFW_TYPE type = static_cast<NNFW_TYPE>(jtype);

  if (g_sess_2_output.find(sess) == g_sess_2_output.end())
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess doesn't exist for TempOutput", __PRETTY_FUNCTION__);
    return false;
  }

  auto &tom = g_sess_2_output.at(sess);
  if (tom.find(index) == tom.end())
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] %d doesn't exist for TempOutput", __PRETTY_FUNCTION__, index);
    return false;
  }

  auto buf = tom.at(index).buf;
  auto bufsize = tom.at(index).bufsize;
  tom.at(index).type = type;

  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d), type(%d), buf(%p), buf_sz(%lu)",
                      __PRETTY_FUNCTION__, index, jtype, buf, bufsize);

  if (nnfw_set_output(sess, index, type, buf, bufsize) == NNFW_STATUS_ERROR)
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
  auto index = static_cast<uint32_t>(jindex);

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
  auto index = static_cast<uint32_t>(jindex);

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

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeGetInputTensorInfo
 * Signature: (JILcom/samsung/onert/NativeSessionWrapper/InternalTensorInfo;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetInputTensorInfo
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jobject jinfo)
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
  auto index = static_cast<uint32_t>(jindex);

  nnfw_tensorinfo tensor_info;
  if (nnfw_input_tensorinfo(sess, index, &tensor_info) == NNFW_STATUS_ERROR) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_input_tensorinfo is failed", __PRETTY_FUNCTION__);
    return false;
  }

  jclass info_cls = env->FindClass("com/samsung/onert/NativeSessionWrapper$InternalTensorInfo");
  if (info_cls == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] java info class is failed", __PRETTY_FUNCTION__);
    return false;
  }

  // type
  jfieldID type_field = env->GetFieldID(info_cls, "type", "I");
  if (type_field == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's type field id is failed", __PRETTY_FUNCTION__);
    return false;
  }
  jint jtype = static_cast<jint>(tensor_info.dtype);
  env->SetIntField(jinfo, type_field, jtype);

  // rank
  jfieldID rank_field = env->GetFieldID(info_cls, "rank", "I");
  if (rank_field == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's rank field id is failed", __PRETTY_FUNCTION__);
    return false;
  }
  jint jrank = tensor_info.rank;
  env->SetIntField(jinfo, rank_field, jrank);

  // shape
  jfieldID shape_field = env->GetFieldID(info_cls, "shape", "[I");
  if (shape_field == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's shape field id is failed", __PRETTY_FUNCTION__);
    return false;
  }
  jintArray jshape = env->NewIntArray(jrank);
  if (jshape == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's shape[] allocation is failed", __PRETTY_FUNCTION__);
    return false;

  }
  env->SetIntArrayRegion(jshape, 0, jrank, (tensor_info.dims));
  env->SetObjectField(jinfo, shape_field, jshape);

  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeGetOutputTensorInfo
 * Signature: (JILcom/samsung/onert/NativeSessionWrapper/InternalTensorInfo;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetOutputTensorInfo
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jobject jinfo)
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
  auto index = static_cast<uint32_t>(jindex);

  nnfw_tensorinfo tensor_info;
  if (nnfw_output_tensorinfo(sess, index, &tensor_info) == NNFW_STATUS_ERROR) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] nnfw_output_tensorinfo is failed", __PRETTY_FUNCTION__);
    return false;
  }

  jclass info_cls = env->FindClass("com/samsung/onert/NativeSessionWrapper$InternalTensorInfo");
  if (info_cls == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] java info class is failed", __PRETTY_FUNCTION__);
    return false;
  }

  // type
  jfieldID type_field = env->GetFieldID(info_cls, "type", "I");
  if (type_field == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's type field id is failed", __PRETTY_FUNCTION__);
    return false;
  }
  jint jtype = static_cast<jint>(tensor_info.dtype);
  env->SetIntField(jinfo, type_field, jtype);

  // rank
  jfieldID rank_field = env->GetFieldID(info_cls, "rank", "I");
  if (rank_field == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's rank field id is failed", __PRETTY_FUNCTION__);
    return false;
  }
  jint jrank = tensor_info.rank;
  env->SetIntField(jinfo, rank_field, jrank);

  // shape
  jfieldID shape_field = env->GetFieldID(info_cls, "shape", "[I");
  if (shape_field == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's shape field id is failed", __PRETTY_FUNCTION__);
    return false;
  }
  jintArray jshape = env->NewIntArray(jrank);
  if (jshape == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        "%s] TensorInfo's shape[] allocation is failed", __PRETTY_FUNCTION__);
    return false;

  }
  env->SetIntArrayRegion(jshape, 0, jrank, (tensor_info.dims));
  env->SetObjectField(jinfo, shape_field, jshape);

  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeNewTempOutputBuf
 * Signature: (JII)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeNewTempOutputBuf
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex, jint jbufsize)
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
  auto index = static_cast<uint32_t>(jindex);

  if (jbufsize < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] bufsize(%d) is wrong", __PRETTY_FUNCTION__, jbufsize);
    return false;
  }
  auto bufsize = static_cast<size_t>(jbufsize);

  if (g_sess_2_output.find(sess) == g_sess_2_output.end())
  {
     TempOutputMap tom;
     tom.emplace(index, TempOutput{new char[bufsize]{}, bufsize, static_cast<NNFW_TYPE>(0)});
     g_sess_2_output.emplace(sess, tom);
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] new TempOutputBuf for #(%u) on buf(%p) bufsize(%lu)",
                        __PRETTY_FUNCTION__, index, tom.at(index).buf, tom.at(index).bufsize);
  }
  else
  {
    auto &tom = g_sess_2_output.at(sess);
    tom.emplace(index, TempOutput{new char[bufsize]{}, bufsize, static_cast<NNFW_TYPE>(0)});
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] new TempOutputBuf for #(%u) on buf(%p) bufsize(%lu)",
                        __PRETTY_FUNCTION__, index, tom.at(index).buf, tom.at(index).bufsize);
  }
  
  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeDeleteTempOutputBuf
 * Signature: (JI)Z
 */
JNIEXPORT jboolean JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeDeleteTempOutputBuf
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex)
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
  auto index = static_cast<size_t>(jindex);

  if (g_sess_2_output.find(sess) == g_sess_2_output.end())
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess doesn't exist for TempOutput", __PRETTY_FUNCTION__);
    return false;
  }

  auto &tom = g_sess_2_output.at(sess);
  if (tom.find(index) == tom.end())
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] %lu doesn't exist for TempOutput", __PRETTY_FUNCTION__, index);
    return false;
  }

  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] delete TempOutputBuf for #(%lu) on buf(%p) bufsize(%lu)",
                      __PRETTY_FUNCTION__, index, tom.at(index).buf, tom.at(index).bufsize);
  delete[] tom.at(index).buf;
  tom.erase(index);
  return true;
}

/*
 * Class:     com_samsung_onert_NativeSessionWrapper
 * Method:    nativeGetOutputBuf
 * Signature: (JI)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_samsung_onert_NativeSessionWrapper_nativeGetOutputBuf
  (JNIEnv *env, jobject thiz, jlong handle, jint jindex)
{
  nnfw_session *sess = reinterpret_cast<nnfw_session *>(handle);
  if (sess == nullptr)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess is null", __PRETTY_FUNCTION__);
    return nullptr;
  }

  if (jindex < 0)
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] index(%d) is wrong", __PRETTY_FUNCTION__, jindex);
    return nullptr;
  }
  auto index = static_cast<size_t>(jindex);

  if (g_sess_2_output.find(sess) == g_sess_2_output.end())
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] sess doesn't exist for TempOutput", __PRETTY_FUNCTION__);
    return nullptr;
  }

  auto &tom = g_sess_2_output.at(sess);
  if (tom.find(index) == tom.end())
  {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s] %ld doesn't exist for TempOutput", __PRETTY_FUNCTION__, index);
    return nullptr;
  }

  auto &to = tom.at(index);
  return env->NewDirectByteBuffer(static_cast<void*>(to.buf),static_cast<jlong>(to.bufsize));
}
