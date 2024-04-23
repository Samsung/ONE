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

#include "nnfw_api_internal.h"
#include "nnfw_version.h"

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

/*
 * Create a new session instance
 *
 * @param session the session to be created
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_create_session(nnfw_session **session) { return nnfw_session::create(session); }

/*
 * Close a session instance
 *
 * @param session the session to be closed
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_close_session(nnfw_session *session)
{
  delete session;
  return NNFW_STATUS_NO_ERROR;
}

/*
 * Load model from nnpackage file or directory
 *
 * @param session nnfw_session loading the given nnpackage file/dir
 * @param package_file_path path to the nnpackage file or unzipped directory to be loaded
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_load_model_from_file(nnfw_session *session, const char *pacakge_file_path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->load_model_from_nnpackage(pacakge_file_path);
}

/*
 * Prepare session to be ready for inference
 * This phase may finalize model compilation, scheduling, and additional settings.
 *
 * @param session the session to be prepared
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_prepare(nnfw_session *session)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->prepare();
}

/*
 * Run inference
 *
 * @param session the session to run inference
 * @return NNFW_STATUS_NO_ERROR if successful
 */
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

/*
 * Set input
 *
 * @param session session to the input is to be set
 * @param index index of input to be set (0-indexed)
 * @param type type of the input
 * @param buffer raw buffer for input
 * @param length size of bytes of input
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */

NNFW_STATUS nnfw_set_input(nnfw_session *session, uint32_t index, NNFW_TYPE type,
                           const void *buffer, size_t length)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input(index, type, buffer, length);
}

/*
 * Set output
 *
 * @param session session from inference output is to be extracted
 * @param index index of output to be set (0-indexed)
 * @param type type of the output
 * @param buffer raw buffer for output
 * @param length size of bytes of output
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */

NNFW_STATUS nnfw_set_output(nnfw_session *session, uint32_t index, NNFW_TYPE type, void *buffer,
                            size_t length)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_output(index, type, buffer, length);
}

/*
 * Get the number of inputs
 *
 * @param[in] session session from input information is to be extracted
 * @param[out] number variable which the number of inputs is put into
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */

NNFW_STATUS nnfw_input_size(nnfw_session *session, uint32_t *number)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->input_size(number);
}

/*
 * Get the number of outputs
 *
 * @param[in] session session from output information is to be extracted
 * @param[out] number variable which the number of outputs is put into
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_output_size(nnfw_session *session, uint32_t *number)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->output_size(number);
}

/*
 * Set the layout of an input
 * @note The input that does not call this has NNFW_LAYOUT_CHANNELS_LAST layout
 *
 * @param[in] session session from inference input is to be extracted
 * @param[in] index   index of input to be set (0-indexed)
 * @param[in] layout  layout to set to target input
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_input_layout(nnfw_session *session, uint32_t index, NNFW_LAYOUT layout)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input_layout(index, layout);
}

/*
 * Set the layout of an output
 * @note The output that does not call this has NNFW_LAYOUT_CHANNELS_LAST layout
 *
 * @param[in] session session from inference output is to be extracted
 * @param[in] index   index of output to be set (0-indexed)
 * @param[in] layout  layout to set to target output
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_output_layout(nnfw_session *session, uint32_t index, NNFW_LAYOUT layout)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_output_layout(index, layout);
}

/*
 * Get i-th input tensor info
 *
 * @param[in] session session from input information is to be extracted
 * @param[in] index index of input
 * @param[out] tensor_info nnfw_tensor_info
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_input_tensorinfo(nnfw_session *session, uint32_t index,
                                  nnfw_tensorinfo *tensor_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->input_tensorinfo(index, tensor_info);
}

/*
 * Get i-th output tensor info
 *
 * @param[in] session session from output information is to be extracted
 * @param[in] index index of output
 * @param[out] tensor_info nnfw_tensor_info
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_output_tensorinfo(nnfw_session *session, uint32_t index,
                                   nnfw_tensorinfo *tensor_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->output_tensorinfo(index, tensor_info);
}

/*
 * Register custom operation
 * @param session session to register this operation
 * @param id operation id
 * @param info registration info ( eval function, etc. )
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_register_custom_op_info(nnfw_session *session, const char *id,
                                         custom_kernel_registration_info *info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->register_custom_operation(id, info->eval_function);
}

NNFW_STATUS nnfw_apply_tensorinfo(nnfw_session *, uint32_t, nnfw_tensorinfo)
{
  return NNFW_STATUS_DEPRECATED_API;
}

NNFW_STATUS nnfw_set_input_tensorinfo(nnfw_session *session, uint32_t index,
                                      const nnfw_tensorinfo *tensor_info)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_input_tensorinfo(index, tensor_info);
}

/*
 * Set available backends
 *
 * @param[in] session session to which a avilable backends are set
 * @param[in] backends available backends on which nnfw uses
 */
NNFW_STATUS nnfw_set_available_backends(nnfw_session *session, const char *backends)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_available_backends(backends);
}

/*
 * Set the operation's backend
 *
 * @param[in] session session to be modified
 * @param[in] op operation to be set
 * @param[in] backend bakcend on which operation run
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_op_backend(nnfw_session *session, const char *op, const char *backend)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_op_backend(op, backend);
}

/*
 * Retrieve uint32 type of nnfw information for given information ID.
 *
 * @param[in] session session to be queried on
 * @param[in] information ID to be queried
 * @param[out] val uint32 value to be returned
 *
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
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

NNFW_STATUS nnfw_load_circle_from_buffer(nnfw_session *session, uint8_t *buffer, size_t size)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->load_circle_from_buffer(buffer, size);
}

NNFW_STATUS nnfw_load_model_from_modelfile(nnfw_session *session, const char *file_path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->load_model_from_modelfile(file_path);
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

NNFW_STATUS nnfw_prepare_pipeline(nnfw_session *session, const char *map_file_path)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->prepare_pipeline(map_file_path);
}

NNFW_STATUS nnfw_push_pipeline_input(nnfw_session *session, void *inputs, void *lengths)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->push_pipeline_input((std::vector<void *> *)inputs,
                                      (std::vector<uint32_t> *)lengths);
}

NNFW_STATUS nnfw_pop_pipeline_output(nnfw_session *session, void *outputs)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->pop_pipeline_output((std::vector<void *> *)outputs);
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
