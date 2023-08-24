/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_TRAIN_OPTIMIZER_OPTIMIZER_HELPERS_H__

namespace nnfw
{
namespace cker
{
namespace train
{

// From tensorflow/core/kernels/training_op_helpers.h

// Returns a borrowed pointer to the mutex for the variable `input` in `ctx`.
//
// If `input` corresponds to a `DT_RESOURCE`-type variable input,
// `*maybe_resource` will be updated to contain the underlying resource, and the
// caller will be responsible for calling `Unref()` on that resource.
template <typename Device, typename T>
mutex *GetTrainingVariableMutex(OpKernelContext *ctx, int input, Var **maybe_resource)
{
  *maybe_resource = nullptr;
  if (ctx->input_dtype(input) == DT_RESOURCE)
  {
    if (LookupResource(ctx, HandleFromInput(ctx, input), maybe_resource).ok())
    {
      return (*maybe_resource)->mu();
    }
    else
    {
      ctx->CtxFailureWithWarning(errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return ctx->input_ref_mutex(input);
}

// MaybeLockVariableInputMutexesInOrder is a helper function to acquire mutexes
// in address order to mitigate deadlock.  Returns a structure that, when
// deleted, will release the acquired mutexes. Safe to pass duplicates - will
// only lock each distinct mutex once. If sparse is true will ensure the
// variable gets switched to copy-on-read mode before trying to acquire the
// locks. If do_lock is false, returns immediately for reference variables. For
// resource variables in copy-on-read-mode it will grab a shared lock if do_lock
// is false, exclusive lock otherwise.  Note that this silently doesn't lock
// mutexes for invalid variable references; in all usages this is followed by
// GetInputTensor which will signal a failure.
template <typename Device, typename T>
VariableInputLockHolder MaybeLockVariableInputMutexesInOrder(OpKernelContext *ctx, bool do_lock,
                                                             bool sparse,
                                                             const std::vector<int> &input_ids)
{
  bool any_resource = false;
  //   for (auto i : input_ids) {
  //     if (ctx->input_dtype(i) == DT_RESOURCE) {
  //       any_resource = true;
  //       break;
  //     }
  //   }
  if (!do_lock && !any_resource)
  {
    return VariableInputLockHolder({}, {}, {});
  }
  std::vector<Var *> vars;
  std::vector<mutex *> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids)
  {
    Var *var;
    mutex *mutex = GetTrainingVariableMutex<Device, T>(ctx, input, &var);
    if (var)
      vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end())
    {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  auto locks = std::make_unique<std::vector<mutex_lock>>();
  auto shared_locks = std::make_unique<std::vector<tf_shared_lock>>();
  locks->reserve(acquire_order.size());

  for (auto acquire : acquire_order)
  {
    mutex *mu = mutexes[acquire];
    if (mu != nullptr)
    {
      if (!sparse || do_lock)
      {
        locks->emplace_back(*mu);
      }
      else
      {
        shared_locks->emplace_back(*mu);
      }
    }
  }
  auto variableInputLock = VariableInputLockHolder(vars, std::move(locks), std::move(shared_locks));
  if (sparse)
  {
    // Enable sparse variables' access.
    // NOTE: This can not be done before the variable input locks are held,
    // because a race condition can happen between this and another thread that
    // turns off some variable's `copy_on_read_mode` after this thread enables
    // sparse access; when a later function sees `copy_on_read_mode` is off, it
    // will try to lock the variable again for updating `copy_on_read_mode` and
    // cause the deadlock, since the variable mutex is non-re-entrant.
    for (auto *var : vars)
    {
      EnsureSparseVariableAccess<Device, T>(ctx, var, /*lock_held=*/true).IgnoreError();
    }
  }
  return variableInputLock;
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPTIMIZER_OPTIMIZER_HELPERS_H__
