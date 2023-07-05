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

// Used for training

/*
 * Loss type for training
 */
typedef enum
{
  /** Categorical CrossEntropy loss */
  NNFW_LOSS_TYPE_CATEGORICAL_CROSSENTROPY = 0,
  NNFW_LOSS_TYPE_MEAN_SQUARED_ERROR = 1,
} NNFW_LOSS_TYPE;

typedef struct nnfw_traininfo
{
  uint32_t batchsize;
  NNFW_LOSS_TYPE loss_type;
  // NOTE It assumes that true buf data has the same tensor info with pred index.
  // NOTE This index is based on nnfw_output_size().
  uint32_t y_pred_index;
} nnfw_traininfo;

typedef enum
{
  NNFW_DATA_TYPE_TRAIN = 0,
  NNFW_DATA_TYPE_VALID = 1,
} NNFW_DATA_TYPE;

typedef struct nnfw_data
{
  int num;
  nnfw_tensorinfo *infos;
  const void *bufs;
} nnfw_data;

NNFW_STATUS nnfw_prepare_train(nnfw_session *session, const nnfw_traininfo *train_info);
// TODO Consider multiple expected output
NNFW_STATUS nnfw_train(nnfw_session *session, int epoch, NNFW_DATA_TYPE dtype,
                       const nnfw_data *dataset);

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

#endif // __NNFW_EXPERIMENTAL_H__
