/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ndk_main.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "tflite/Assert.h"
#include "tflite/Session.h"
#include "tflite/InterpreterSession.h"
#include "tflite/NNAPISession.h"
#include "tflite/ext/kernels/register.h"

#include "misc/benchmark.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>

#include <cassert>
#include <chrono>
#include <sstream>

#include <android/log.h>

using namespace tflite;
using namespace tflite::ops::builtin;

static StderrReporter error_reporter;

static std::unique_ptr<FlatBufferModel> model;

inline void setText(JNIEnv *env, jobject thisObj, const std::string &message)
{
  jclass thisClass = env->GetObjectClass(thisObj);
  jmethodID setTextMethod = env->GetMethodID(thisClass, "setText", "(Ljava/lang/String;)V");

  assert(setTextMethod != nullptr);

  env->CallVoidMethod(thisObj, setTextMethod, env->NewStringUTF(message.c_str()));
}

inline void setTitle(JNIEnv *env, jobject thisObj, const std::string &message)
{
  jclass thisClass = env->GetObjectClass(thisObj);
  jmethodID setTextMethod = env->GetMethodID(thisClass, "setTitle", "(Ljava/lang/String;)V");

  assert(setTextMethod != nullptr);

  env->CallVoidMethod(thisObj, setTextMethod, env->NewStringUTF(message.c_str()));

  // Clear message
  setText(env, thisObj, "");
}

inline void setText(JNIEnv *env, jobject thisObj, const std::stringstream &ss)
{
  setText(env, thisObj, ss.str());
}

inline std::unique_ptr<FlatBufferModel> loadModel(JNIEnv *env, jobject thisObj,
                                                  jobject model_buffer)
{
  const char *model_base = static_cast<char *>(env->GetDirectBufferAddress(model_buffer));
  jlong model_size = env->GetDirectBufferCapacity(model_buffer);

  return FlatBufferModel::BuildFromBuffer(model_base, static_cast<size_t>(model_size),
                                          &error_reporter);
}

struct Activity
{
  virtual ~Activity() = default;

  virtual void prepare(void) const = 0;
  virtual void run(void) const = 0;
  virtual void teardown(void) const = 0;
};

struct LiteActivity final : public Activity
{
public:
  LiteActivity(nnfw::tflite::Session &sess) : _sess(sess)
  {
    // DO NOTHING
  }

public:
  void prepare(void) const override { _sess.prepare(); }
  void run(void) const override { _sess.run(); }
  void teardown(void) const override { _sess.teardown(); }

private:
  nnfw::tflite::Session &_sess;
};

struct SimpleActivity final : public Activity
{
public:
  SimpleActivity(const std::function<void(void)> &fn) : _fn{fn}
  {
    // DO NOTHING
  }

public:
  void prepare(void) const override {}
  void run(void) const override { _fn(); }
  void teardown(void) const override {}

private:
  std::function<void(void)> _fn;
};

inline void runBenchmark(JNIEnv *env, jobject thisObj, Activity &act)
{
  auto runTrial = [&](void) {
    std::chrono::milliseconds elapsed(0);

    act.prepare();
    nnfw::misc::benchmark::measure(elapsed) << [&](void) { act.run(); };
    act.teardown();

    return elapsed;
  };

  // Warm-up
  for (uint32_t n = 0; n < 3; ++n)
  {
    auto elapsed = runTrial();

    std::stringstream ss;
    ss << "Warm-up #" << n << "  takes " << elapsed.count() << "ms" << std::endl;
    setText(env, thisObj, ss);
  }

  // Measure
  using namespace boost::accumulators;

  accumulator_set<double, stats<tag::mean, tag::min, tag::max>> acc;

  for (uint32_t n = 0; n < 100; ++n)
  {
    auto elapsed = runTrial();

    std::stringstream ss;
    ss << "Iteration #" << n << " takes " << elapsed.count() << "ms" << std::endl;
    setText(env, thisObj, ss);

    acc(elapsed.count());
  }

  std::stringstream ss;
  ss << "Average is " << mean(acc) << "ms" << std::endl;
  ss << "Min is " << min(acc) << "ms" << std::endl;
  ss << "Max is " << max(acc) << "ms" << std::endl;
  setText(env, thisObj, ss);
}

JNIEXPORT void JNICALL Java_com_ndk_tflbench_MainActivity_runInterpreterBenchmark(
  JNIEnv *env, jobject thisObj, jobject model_buffer)
{
  setTitle(env, thisObj, "Running Interpreter Benchmark");

  auto model = loadModel(env, thisObj, model_buffer);
  assert(model != nullptr);

  nnfw::tflite::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;

  TFLITE_ENSURE(builder(&interpreter));

  interpreter->SetNumThreads(-1);

  nnfw::tflite::InterpreterSession sess(interpreter.get());
  LiteActivity act{sess};
  runBenchmark(env, thisObj, act);
}

static void runNNAPIBenchmark(JNIEnv *env, jobject thisObj, jobject model_buffer)
{
  auto model = loadModel(env, thisObj, model_buffer);
  assert(model != nullptr);

  nnfw::tflite::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;

  TFLITE_ENSURE(builder(&interpreter));

  nnfw::tflite::NNAPISession sess(interpreter.get());
  LiteActivity act{sess};
  runBenchmark(env, thisObj, act);
}

JNIEXPORT void JNICALL Java_com_ndk_tflbench_MainActivity_runNNAPIBenchmark(JNIEnv *env,
                                                                            jobject thisObj,
                                                                            jobject model_buffer)
{
  setTitle(env, thisObj, "Running NNAPI Benchmark");

  try
  {
    runNNAPIBenchmark(env, thisObj, model_buffer);
  }
  catch (const std::exception &ex)
  {
    std::stringstream ss;
    ss << "Caught an exception " << ex.what();
    setText(env, thisObj, ss);
  }
}

JNIEXPORT jstring JNICALL Java_com_ndk_tflbench_MainActivity_getModelName(JNIEnv *env,
                                                                          jobject thisObj)
{
  return env->NewStringUTF(MODEL_NAME);
}

#define TF_ENSURE(e)                               \
  {                                                \
    if (!(e).ok())                                 \
    {                                              \
      throw std::runtime_error{"'" #e "' FAILED"}; \
    }                                              \
  }
