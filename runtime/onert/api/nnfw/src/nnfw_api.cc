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

#include "nnfw.h"
#include "nnfw_experimental.h"
#include "nnfw_version.h"

#include "nnfw_session.h"

// Double-check enum value changes

#define STATIC_ASSERT_ENUM_CHECK(ENUM, VAL) static_assert((ENUM) == (VAL), #ENUM " has changed")

STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_FLOAT32, 0);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_INT32, 1);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_QUANT8_ASYMM, 2);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_BOOL, 3);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_UINT8, 4);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_INT64, 5);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED, 6);
STATIC_ASSERT_ENUM_CHECK(NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED, 7);

STATIC_ASSERT_ENUM_CHECK(NNFW_STATUS_NO_ERROR, 0);
STATIC_ASSERT_ENUM_CHECK(NNFW_STATUS_ERROR, 1);
STATIC_ASSERT_ENUM_CHECK(NNFW_STATUS_UNEXPECTED_NULL, 2);
STATIC_ASSERT_ENUM_CHECK(NNFW_STATUS_INVALID_STATE, 3);
STATIC_ASSERT_ENUM_CHECK(NNFW_STATUS_OUT_OF_MEMORY, 4);
STATIC_ASSERT_ENUM_CHECK(NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE, 5);

STATIC_ASSERT_ENUM_CHECK(NNFW_LAYOUT_NONE, 0);
STATIC_ASSERT_ENUM_CHECK(NNFW_LAYOUT_CHANNELS_LAST, 1);
STATIC_ASSERT_ENUM_CHECK(NNFW_LAYOUT_CHANNELS_FIRST, 2);

STATIC_ASSERT_ENUM_CHECK(NNFW_INFO_ID_VERSION, 0);

#undef STATIC_ASSERT_ENUM_CHECK

#define NNFW_RETURN_ERROR_IF_NULL(p)      \
  do                                      \
  {                                       \
    if ((p) == NULL)                      \
      return NNFW_STATUS_UNEXPECTED_NULL; \
  } while (0)

NNFW_STATUS nnfw_create_session(nnfw_session **session) { return nnfw_session::create(session); }

NNFW_STATUS nnfw_close_session(nnfw_session *session)
{
  delete session;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_load_model_from_file(nnfw_session *session, const char *path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->load_model_from_path(path);
}

NNFW_STATUS nnfw_prepare(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->prepare();
}

NNFW_STATUS nnfw_run(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->run();
}

NNFW_STATUS nnfw_run_async(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->run_async();
}

NNFW_STATUS nnfw_await(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->await();
}

NNFW_STATUS nnfw_set_input(nnfw_session *session, uint32_t index, NNFW_TYPE type,
                           const void *buffer, size_t length)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input(index, type, buffer, length);
}

NNFW_STATUS nnfw_set_output(nnfw_session *session, uint32_t index, NNFW_TYPE type, void *buffer,
                            size_t length)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_output(index, type, buffer, length);
}

NNFW_STATUS nnfw_input_size(nnfw_session *session, uint32_t *number)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->input_size(number);
}

NNFW_STATUS nnfw_output_size(nnfw_session *session, uint32_t *number)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->output_size(number);
}

NNFW_STATUS nnfw_set_input_layout(nnfw_session *session, uint32_t index, NNFW_LAYOUT layout)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input_layout(index, layout);
}

NNFW_STATUS nnfw_set_output_layout(nnfw_session *session, uint32_t index, NNFW_LAYOUT layout)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_output_layout(index, layout);
}

NNFW_STATUS nnfw_set_input_type(nnfw_session *session, uint32_t index, NNFW_TYPE type)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input_type(index, type);
}

NNFW_STATUS nnfw_set_output_type(nnfw_session *session, uint32_t index, NNFW_TYPE type)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_output_type(index, type);
}

NNFW_STATUS nnfw_input_tensorinfo(nnfw_session *session, uint32_t index,
                                  nnfw_tensorinfo *tensor_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->input_tensorinfo(index, tensor_info);
}

NNFW_STATUS nnfw_output_tensorinfo(nnfw_session *session, uint32_t index,
                                   nnfw_tensorinfo *tensor_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->output_tensorinfo(index, tensor_info);
}

NNFW_STATUS nnfw_register_custom_op_info(nnfw_session *session, const char *id,
                                         custom_kernel_registration_info *info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->register_custom_operation(id, info->eval_function);
}

NNFW_STATUS nnfw_apply_tensorinfo(nnfw_session *, uint32_t, nnfw_tensorinfo)
{
  return nnfw_session::deprecated("nnfw_apply_tensorinfo: Deprecated");
}

NNFW_STATUS nnfw_set_input_tensorinfo(nnfw_session *session, uint32_t index,
                                      const nnfw_tensorinfo *tensor_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input_tensorinfo(index, tensor_info);
}

NNFW_STATUS nnfw_set_available_backends(nnfw_session *session, const char *backends)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_available_backends(backends);
}

NNFW_STATUS nnfw_set_op_backend(nnfw_session *, const char *, const char *)
{
  return nnfw_session::deprecated("nnfw_set_op_backend: Deprecated");
}

NNFW_STATUS nnfw_query_info_u32(nnfw_session *session, NNFW_INFO_ID id, uint32_t *val)
{
  (void)session;
  switch (id)
  {
    case NNFW_INFO_ID_VERSION:
      if (val)
      {
        *val = NNFW_VERSION;
        return NNFW_STATUS_NO_ERROR;
      }
      break;
    default:
      return NNFW_STATUS_ERROR;
  }
  // It should not be reached.
  return NNFW_STATUS_ERROR;
}

NNFW_STATUS nnfw_input_tensorindex(nnfw_session *session, const char *tensorname, uint32_t *index)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->input_tensorindex(tensorname, index);
}

NNFW_STATUS nnfw_output_tensorindex(nnfw_session *session, const char *tensorname, uint32_t *index)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->output_tensorindex(tensorname, index);
}

NNFW_STATUS nnfw_set_backends_per_operation(nnfw_session *session, const char *backend_settings)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_backends_per_operation(backend_settings);
}

NNFW_STATUS nnfw_prepare_pipeline(nnfw_session *, const char *)
{
  return nnfw_session::deprecated("nnfw_prepare_pipeline: Deprecated");
}

NNFW_STATUS nnfw_push_pipeline_input(nnfw_session *, void *, void *)
{
  return nnfw_session::deprecated("nnfw_push_pipeline_input: Deprecated");
}

NNFW_STATUS nnfw_pop_pipeline_output(nnfw_session *, void *)
{
  return nnfw_session::deprecated("nnfw_pop_pipeline_output: Deprecated");
}

NNFW_STATUS nnfw_set_workspace(nnfw_session *session, const char *dir)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_workspace(dir);
}

// Training

NNFW_STATUS nnfw_train_get_traininfo(nnfw_session *session, nnfw_train_info *info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_get_traininfo(info);
}

NNFW_STATUS nnfw_train_set_traininfo(nnfw_session *session, const nnfw_train_info *info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_set_traininfo(info);
}

NNFW_STATUS nnfw_train_prepare(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_prepare();
}

NNFW_STATUS nnfw_train_input_tensorinfo(nnfw_session *session, uint32_t index,
                                        nnfw_tensorinfo *info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_input_tensorinfo(index, info);
}

NNFW_STATUS nnfw_train_expected_tensorinfo(nnfw_session *session, uint32_t index,
                                           nnfw_tensorinfo *info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_expected_tensorinfo(index, info);
}

NNFW_STATUS nnfw_train_set_input(nnfw_session *session, uint32_t index, const void *input,
                                 const nnfw_tensorinfo *input_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_set_input(index, input, input_info);
}

NNFW_STATUS nnfw_train_set_expected(nnfw_session *session, uint32_t index, const void *expected,
                                    const nnfw_tensorinfo *expected_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_set_expected(index, expected, expected_info);
}

NNFW_STATUS nnfw_train_set_output(nnfw_session *session, uint32_t index, NNFW_TYPE type,
                                  void *buffer, size_t length)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_set_output(index, type, buffer, length);
}

NNFW_STATUS nnfw_train(nnfw_session *session, bool update_weights)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_run(update_weights);
}

NNFW_STATUS nnfw_train_get_loss(nnfw_session *session, uint32_t index, float *loss)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_get_loss(index, loss);
}

NNFW_STATUS nnfw_train_export_circle(nnfw_session *session, const char *path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_export_circle(path);
}

NNFW_STATUS nnfw_train_import_checkpoint(nnfw_session *session, const char *path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_import_checkpoint(path);
}

NNFW_STATUS nnfw_train_export_checkpoint(nnfw_session *session, const char *path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->train_export_checkpoint(path);
}

// Quantization

NNFW_STATUS nnfw_set_quantization_type(nnfw_session *session, NNFW_QUANTIZE_TYPE qtype)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_quantization_type(qtype);
}

NNFW_STATUS nnfw_set_quantized_model_path(nnfw_session *session, const char *path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_quantized_model_path(path);
}

NNFW_STATUS nnfw_quantize(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->quantize();
}

NNFW_STATUS nnfw_set_codegen_model_path(nnfw_session *session, const char *path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_codegen_model_path(path);
}

NNFW_STATUS nnfw_codegen(nnfw_session *session, const char *target, NNFW_CODEGEN_PREF pref)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->codegen(target, pref);
}

NNFW_STATUS nnfw_set_odc_param_minmax_records_count(nnfw_session *session, int minmax_records_count)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_odc_param_minmax_records_count(minmax_records_count);
}

NNFW_STATUS nnfw_odc_delete_minmax_file(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->delete_odc_minmax_file();
}

NNFW_STATUS nnfw_run_with_auto_compilation(nnfw_session *session, const char *target,
                                           NNFW_CODEGEN_PREF pref)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->run_with_auto_compilation(target, pref);
}

// Configuration

NNFW_STATUS nnfw_set_prepare_config(nnfw_session *session, const NNFW_PREPARE_CONFIG key,
                                    const char *value)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_prepare_config(key, value);
}

NNFW_STATUS nnfw_reset_prepare_config(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->reset_prepare_config();
}

NNFW_STATUS nnfw_set_execute_config(nnfw_session *session, const NNFW_RUN_CONFIG key,
                                    const char *value)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_execute_config(key, value);
}

NNFW_STATUS nnfw_reset_execute_config(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->reset_execute_config();
}
