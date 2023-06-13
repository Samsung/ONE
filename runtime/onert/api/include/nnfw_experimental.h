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

#ifndef __NNFW_EXPERIMENTAL_H__
#define __NNFW_EXPERIMENTAL_H__

#include "nnfw.h"

#ifdef __cplusplus
extern "C" {
#endif

// Used for custom kernel development

/*
 * operand type, used only for custom operations
 */
typedef struct
{
  nnfw_tensorinfo type;
  void *allocation;
} nnfw_operand;

/*
 * Used as input to custom operation eval function
 */
typedef struct
{
  size_t ninputs;
  nnfw_operand *inputs;

  size_t noutputs;
  nnfw_operand *outputs;
} nnfw_custom_kernel_params;

/*
 * Custom kernel evaluation function
 *
 * param[in] params custom operation parameters
 * param[in] userdata pointer to user-specified buffer( kernel instance specific )
 */
typedef void (*nnfw_custom_eval)(nnfw_custom_kernel_params *params, char *userdata,
                                 size_t userdata_size);

/*
 * custom operation registration info
 */
typedef struct
{
  nnfw_custom_eval eval_function;
} custom_kernel_registration_info;

NNFW_STATUS nnfw_register_custom_op_info(nnfw_session *session, const char *id,
                                         custom_kernel_registration_info *info);

/**
 * @brief Get the input tensor index by name
 *
 * This function finds an input tensor of the given name.
 * If found, the index value is set to the address that @c index points to, and returns
 * @c NNFW_STATUS_NO_ERROR. Otherwise, @c index is unchanged and returns @c NNFW_STATUS_ERROR .
 *
 * @note If two or more input tensors are of the same name, the one with the lowest index is always
 *       returned.
 *
 * @param[in]  session    the session object
 * @param[in]  tensorname the name of the tensor to find, a null terminated char pointer string
 * @param[out] index      the index to be ret
 * @return     @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_input_tensorindex(nnfw_session *session, const char *tensorname, uint32_t *index);

/**
 * @brief Get the input tensor index by name
 *
 * This function finds an input tensor of the given name.
 * If found, the index value is set to the address that @c index points to, and returns
 * @c NNFW_STATUS_NO_ERROR. Otherwise, @c index is unchanged and returns @c NNFW_STATUS_ERROR .
 *
 * @note If two or more input tensors are of the same name, the one with the lowest index is always
 *       returned.
 *
 * @param[in]  session    the session object
 * @param[in]  tensorname the name of the tensor to find, a null terminated char pointer string
 * @param[out] index      the index to be ret
 * @return     @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_output_tensorindex(nnfw_session *session, const char *tensorname, uint32_t *index);

/**
 * @brief Set the backend for each operation in the session
 *
 * This function assigns backends (acl_cl, acl_neon, cpu) to each operation in the session.
 * If successful,the function returns @c NNFW_STATUS_NO_ERROR. Otherwise, the function returns
 * @c NNFW_STATUS_ERROR.
 *
 * @note The argument specifying backends must be in the format
 *       "OP_BACKEND_MAP=\"0=acl_cl;1=cpu;2=acl_cl\"".
 *
 * @param[in]  session          the session object
 * @param[in]  backend_settings String containing backend assignments indexed by operation sequence
 * @return     @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_backends_per_operation(nnfw_session *session, const char *backend_settings);

/*
 * Prepare session to be ready for inference
 * This phase may finalize model compilation, scheduling, and additional settings.
 *
 * @param session the session to be prepared
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_prepare_pipeline(nnfw_session *session, const char *map_file_path = nullptr);

/**
 * @brief     Set input buffer
 *
 * This function must be called after {@link nnfw_prepare_pipeline}, \p inputs given to this
 * function can be reused for many inferences. \p lengths must be greater or equal than the operand
 * requires. if you give empty \p inputs to this function, then this function will join all threads.
 *
 * @param[in] session Session to the input is to be set
 * @param[in] inputs  Raw buffers for input, it must be \p std::vector<void *> type pointer for
 * multiple input model
 * @param[in] lengths Size of bytes of input buffers, it must be \p std::vector<uint32_t> type
 * pointer for multiple input model
 *
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_push_pipeline_input(nnfw_session *session, void *inputs, void *lengths);

/**
 * @brief       Get last outputs of partitioned model in session
 *
 * This function must be called after {@link nnfw_prepare_pipeline}, \p outputs given to this
 * function must be cleared for memory management.
 *
 * @param[in]   session Session from last outputs is to be extracted
 * @param[out]  outputs Raw buffer for outputs, it must be \p std::vector<void *> type pointer for
 * multiple output model
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_pop_pipeline_output(nnfw_session *session, void *outputs);

/**
 *  Training C APIs
 *
 * Training APIs are designed to be used in the following order for training
 * 1. nnfw_set_traininfo
 * 2. nnfw_prepare_train
 * 3. nnfw_set_traininput
 * 4. nnfw_set_trainoutput (optional if we use nnfw_get_loss)
 * 5. nnfw_train
 * 6. nnfw_get_loss (optional)
 *
 * If you want to inference after training with the same session, you can use the following order
 * 1. nnfw_set_input
 * 2. nnfw_set_output
 * 3. nnfw_run
 */

//////////////////////////////////////////////
// Essential APIs for training
//////////////////////////////////////////////

/**
 * @brief Loss type
 */
typedef enum
{
  /** Categorical CrossEntropy loss */
  NNFW_LOSS_TYPE_CATEGORICAL_CROSSENTROPY = 0,

} NNFW_LOSS_TYPE;

/**
 * @brief Training information to prepare training
 */
typedef struct nnfw_train_info
{
  /** Loss type */
  NNFW_LOSS_TYPE loss_type;
  /** Learning rate */
  float learning_rate;
  /** Batch size */
  uint32_t batch_size;
} nnfw_train_info;

/**
 * @brief Set training information before prepare training
 * @note  This function must be called before {@link nnfw_prepare_train} to set training information
 *        This function may deprecated if we can get training information from model (not planned)
 *        This function may be used after {@link nnfw_prepare_pipeline}
 *        to support dynamic training information (not planned)
 *
 * @param[in] session The session to be prepared for training
 * @param[in] info    Training information
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_traininfo(nnfw_session *session, nnfw_train_info info);

/**
 * @brief Prepare session to be ready for training
 *
 * @param[in] session The session to be prepared for training
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_prepare_train(nnfw_session *session);

/**
 * @brief Set training inputs and expected model outputs
 *
 * @param[in] session         The session to be set training inputs and expected model outputs
 * @param[in] inputs          The input buffers for training
 * @param[in] input_infos     The shape and type of input buffers
 * @param[in] expecteds       The expected model outputs for training
 * @param[in] expected_infos  The shape and type of expected model outputs
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_traininput(nnfw_session *session, void **inputs, nnfw_tensorinfo *input_infos,
                                void **expecteds, nnfw_tensorinfo *expected_infos);

/**
 * @brief Set training outputs
 *        Training outputs may be used to get loss value after training
 *        It may be deprecated if we allow only @{link nnfw_get_loss} to get loss value
 *
 * @param[in] session       The session to be set training outputs
 * @param[in] outputs       The output buffers for training
 * @param[in] output_infos  The shape and type of output buffers
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_trainoutput(nnfw_session *session, void **outputs,
                                 nnfw_tensorinfo *output_infos);

/**
 * @brief Train the model
 *
 * @param[in] session The session to be trained
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train(nnfw_session *session);

/**
 * @brief Get loss value after training
 *
 * @param[in]   session The session to get loss value
 * @param[out]  loss    The loss value
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_loss(nnfw_session *session, float *loss);

//////////////////////////////////////////////
// Optional APIs for training
//////////////////////////////////////////////

/**
 * @brief Get the number of training model inputs
 *        This function should be called after {@link nnfw_prepare_train}
 *
 * @param[in]   session The session to get the number of training model inputs
 * @param[out]  count   The number of training model inputs
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_traininput_count(nnfw_session *session, uint32_t *count);

/**
 * @brief Get the training model input information
 *
 * @param[in]   session The session to get the training model input information
 * @param[in]   index   The index of training model input
 * @param[out]  info    The shape and type of training model input
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_traininput_tensorinfo(nnfw_session *session, uint32_t index,
                                           nnfw_tensorinfo *info);

/**
 * @brief Get the number of training model expected outputs
 *        This function should be called after {@link nnfw_prepare_train}
 *
 * @param[in]   session The session to get the number of training model expected outputs
 * @param[out]  count   The number of training model expected outputs
 */
NNFW_STATUS nnfw_get_trainexpected_count(nnfw_session *session, uint32_t *count);

/**
 * @brief Get the training model expected output information
 *
 * @param[in]   session The session to get the training model expected output information
 * @param[in]   index   The index of training model expected output
 * @param[out]  info    The shape and type of training model expected output
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_get_trainexpected_tensorinfo(nnfw_session *session, uint32_t index,
                                              nnfw_tensorinfo *info);

/**
 * @brief Export inference model
 *        This function should be called after {@link nnfw_train}
 *
 * @param[in] session The session to export inference model
 * @param[in] path    The path to export inference model
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_export_inference_model(nnfw_session *session, const char *path);

#ifdef __cplusplus
}
#endif

#endif // __NNFW_EXPERIMENTAL_H__
