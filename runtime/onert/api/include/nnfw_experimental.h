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
 * 1. nnfw_train_prepare
 * 2. nnfw_train_set_input, nnfw_train_set_expected for inputs & expected outputs
 * 3. nnfw_train
 * 4. nnfw_train_get_loss
 *
 * If you want to inference after training with the same session, you can use the following order
 * 1. nnfw_set_input
 * 2. nnfw_set_output
 * 3. nnfw_run
 */

//////////////////////////////////////////////
// Essential APIs for training
//////////////////////////////////////////////
typedef enum
{
  NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR = 0,
  NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY = 1,
} NNFW_TRAIN_LOSS;

/**
 * @brief Training information to prepare training
 * @todo  Add more training information
 *        (e.g. optimizer, loss function, ...)
 */
typedef struct nnfw_train_info
{
  /** Learning rate */
  float learning_rate = 0.001f;
  /** Batch size */
  uint32_t batch_size = 1;
  /** loss type */
  NNFW_TRAIN_LOSS loss = NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR;
} nnfw_train_info;

/**
 * @brief Prepare session to be ready for training
 * @note  The session will be entered into training mode
 *
 * @param[in] session The session to be prepared for training
 * @param[in] info    Training information.
 *                    If info is nullptr, it will not change training information.
 *                    If it is nullptr and model has not training information,
 *                    it will use default training information.
 *                    Default training information is {learning_rate = 0.001f, batch_size = 1}
 *
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_prepare(nnfw_session *session, const nnfw_train_info *info);

/**
 * @brief Set training input
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param[in] session     The session to be set training inputs and expected model outputs
 * @param[in] index       The index of training input
 * @param[in] input       The input buffers for training
 * @param[in] input_info  The shape and type of input buffer
 *                        If it is nullptr, it will not change shape and batch size
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_set_input(nnfw_session *session, uint32_t index, const void *input,
                                 const nnfw_tensorinfo *input_info);

/**
 * @brief Set training expected output
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param session       The session to be set training inputs and expected model outputs
 * @param index         The index of training expected output
 * @param expected      The expected buffers for training
 * @param expected_info The shape and type of expected buffer
 *                      If it is nullptr, it will not change shape and batch size
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_set_expected(nnfw_session *session, uint32_t index, const void *expected,
                                    const nnfw_tensorinfo *expected_info);

/**
 * @brief Train the model
 * @note  This function should be called after {@link nnfw_train_set_input} and
 *        {@link nnfw_train_set_expected} for each input and expected output
 *
 * @param[in] session The session to be trained
 * @param[in] update_weights If true, update weights of the model
 *                           If false, do not update weights of the model (for validation)
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train(nnfw_session *session, bool update_weights);

/**
 * @brief Get loss value for expected output
 * @note  This function should be called after {@link nnfw_train}
 *
 * @param[in]   session The session to get loss value
 * @param[in]   index   The index of loss value [0, number of expected outputs)
 * @param[out]  loss    The loss value
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_get_loss(nnfw_session *session, uint32_t index, float *loss);

/**
 * @brief Export circle model
 * @note  This function should be called on training mode
 *        This function should be called after {@link nnfw_train}
 *
 * @param[in] session The session to export inference model
 * @param[in] path    The path to export inference model
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_export_circle(nnfw_session *session, const char *path);

//////////////////////////////////////////////
// Optional APIs for training
//////////////////////////////////////////////

/**
 * @brief Get the training model input information
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param[in]   session The session to get the training model input information
 * @param[in]   index   The index of training model input
 * @param[out]  info    The shape and type of training model input
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_input_tensorinfo(nnfw_session *session, uint32_t index,
                                        nnfw_tensorinfo *info);

/**
 * @brief Get the training model expected output information
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param[in]   session The session to get the training model expected output information
 * @param[in]   index   The index of training model expected output
 * @param[out]  info    The shape and type of training model expected output
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_expected_tensorinfo(nnfw_session *session, uint32_t index,
                                           nnfw_tensorinfo *info);

//////////////////////////////////////////////
// Not planned to be implemented
//////////////////////////////////////////////

/**
 * @brief Convert between training mode and inference mode
 * @note  This function should be called after {@link nnfw_train} or {@link nnfw_prepare}
 *
 * @param[in] session The session to convert training mode to inference mode
 * @param[in] train   If false, convert training model to inference model
 *                    If true, convert inference model to training model
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
// NNFW_STATUS nnfw_set_training_mode(nnfw_session *session, bool train);

/**
 * @brief Set training information after prepare training
 * @note  This function may be used after {@link nnfw_train_prepare}
 *
 * @param[in] session The session prepared for training
 * @param[in] info    Training information
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
// NNFW_STATUS nnfw_train_set_traininfo(nnfw_session *session, const nnfw_train_info info);

#ifdef __cplusplus
}
#endif

#endif // __NNFW_EXPERIMENTAL_H__
