#include "TFRecognizeApp.h"
#include "mbed_mem_trace.h"

using namespace Teddy;
using namespace AI;
using namespace Math;
using namespace std;

static int16_t tmpbuf[WINDOW_SIZE];
static float feature_buffer[kFeatureBufferSize];
size_t bytes_allocated = 0;

void mem_trace_callback(uint8_t op, void *res, void *caller, ...)
{
  va_list va;
  size_t temp_s1, temp_s2;
  void *temp_ptr;

  va_start(va, caller);
  switch (op)
  {
  case MBED_MEM_TRACE_MALLOC:
    temp_s1 = va_arg(va, size_t);
    bytes_allocated += temp_s1;
    // printf(MBED_MEM_DEFAULT_TRACER_PREFIX "m:%p;%p-%u\n", res, caller, temp_s1);
    break;

  case MBED_MEM_TRACE_REALLOC:
    temp_ptr = va_arg(va, void *);
    temp_s1 = va_arg(va, size_t);
    // printf(MBED_MEM_DEFAULT_TRACER_PREFIX "r:%p;%p-%p;%u\n", res, caller, temp_ptr, temp_s1);
    break;

  case MBED_MEM_TRACE_CALLOC:
    temp_s1 = va_arg(va, size_t);
    temp_s2 = va_arg(va, size_t);
    // printf(MBED_MEM_DEFAULT_TRACER_PREFIX "c:%p;%p-%u;%u\n", res, caller, temp_s1, temp_s2);
    break;

  case MBED_MEM_TRACE_FREE:
    temp_ptr = va_arg(va, void *);
    // printf(MBED_MEM_DEFAULT_TRACER_PREFIX "f:%p;%p-%p\n", res, caller, temp_ptr);
    break;

  default:
    printf("?\n");
  }
  va_end(va);
}
TFRecognizeApp::TFRecognizeApp(const osPriority_t thread_priority,
                               const char *name, Mail<mail_t, 8> &mail_box_rx,
                               QueueMMPC_t &mail_box_tx,
                               const tflite::Model *model, ILogger &logger,
                               uint8_t *tensor_arena, uint8_t *recording_buffer)
    : ICore(thread_priority, name),
      _mail_box_rx(mail_box_rx),
      _mail_box_tx(mail_box_tx),
      _model(model),
      logger(logger),
      _tensor_arena(tensor_arena)
{
  // int sampFreq = 5333, int nCep = 20, int winLength = 50,int frameShift = 50,
  //   int numFilt = 12, double lf = 50,double hf = 2666
  _mfcc = new MFCC(kMFCCSamplerate, kMFCCNumberOfCepstral, kMFCCWinLength,
                   kMFCCFrameShift, kMFCCNumberOfFilters, kMFCCLowestFrequency,
                   kMFCCHighestFrequency);

  //   for (int i = 0; i <= demo5333_raw_size - 2 * WINDOW_SIZE; i += 2 *
  //   HOP_SIZE) {
  //     memcpy(tmpbuf, demo5333_raw + i, WINDOW_SIZE * 2);
  //     // _mfcc->processFrame<v_f_t>(tmpbuf, WINDOW_SIZE, _feature_vector);
  //     auto outstr = _mfcc->processFrame(tmpbuf, WINDOW_SIZE);
  //     logger.printf("%s\n", outstr.c_str());
  //   }
  //   for (int i = 0; i < kFeatureBufferSize; i++) {
  //     feature_buffer[i] = _feature_vector[i];
  //   }
  _feature_vector.clear();
  // extern const long int demo5333_raw_size;
  // extern const unsigned char demo5333_raw[];
  _is_ready = true;
  //   _phrase_buf = new std::vector<int16_t>();
  //   _phrase_buf->reserve(kPhraseBufferSize);
  //   _phrase_buf->assign(0,kPhraseBufferSize);
  _phrase_buf = reinterpret_cast<int16_t *>(recording_buffer);
  start();
  // Determine whether a command was recognized based on the output of inference
  //   const char* found_command = nullptr;
  //   uint8_t score = 0;
  //   bool is_new_command = false;
}
TFRecognizeApp::~TFRecognizeApp() { delete _mfcc; }

void TFRecognizeApp::process_frame(std::vector<int16_t> vec)
{
  mail_t *mail;
  if (_is_ready)
  {
    mail = _mail_box_rx.alloc();
    mail->type = MessageType_t::NEW_AUDIO_FRAME_EVENT;
    mail->frame = vec;
    _mail_box_rx.put(mail);
  }
}
void TFRecognizeApp::send_listen_event(Teddy::ListenNode::Hotword hotword)
{
  // mail_t *mail;
  // if (!_mail_box_tx.full()) {
  //   logger(TraceLevel_t::DEBUG) << "TFRecognizeApp: " << hotword << '\n';
  //   mail_t *mail = _mail_box_tx.alloc();
  //   mail->type = MessageType_t::LISTEN_EVENT;
  //   mail->listen_event.type =
  //       Teddy::ListenNode::TriggerType::TriggerType_Hotword;
  //   mail->listen_event.data = hotword;
  //   _mail_box_tx.put(mail);
  // }
  mail_t mail;
  logger(TraceLevel_t::DEBUG) << "TFRecognizeApp: " << hotword << '\n';
  mail.type = MessageType_t::LISTEN_EVENT;
  mail.listen_event.type = Teddy::ListenNode::TriggerType::TriggerType_Hotword;
  mail.listen_event.data = hotword;
  _mail_box_tx.push(mail);
}
void TFRecognizeApp::setup(void)
{
  // logger(TraceLevel_t::DEBUG)
  //     << "Setup TFRecognizeApp " << ThisThread::get_id() << '\n';
}
void TFRecognizeApp::loop(void)
{
  Timer t;
  static bool is_recording_started = false;
  // logger(TraceLevel_t::DEBUG)
  //     << "Loop TFRecognizeApp " << ThisThread::get_id() << '\n';
  mbed_mem_trace_set_callback(mem_trace_callback);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  static tflite::MicroOpResolver<7> micro_op_resolver;
  // micro_op_resolver.AddBuiltin(
  //     tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  //     tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
  //                              tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
  //                              tflite::ops::micro::Register_SOFTMAX());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MUL,
  //                              tflite::ops::micro::Register_MUL());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_ADD,
  //                              tflite::ops::micro::Register_ADD());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
  //                              tflite::ops::micro::Register_MAX_POOL_2D());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      _model, micro_op_resolver, _tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  model_input = interpreter->input(0);
  // logger.warning("TFLite micro interpreter\n");

  // logger.debug("model_no_quant.tflite\n");

  // logger.info("model_input->dims->size %d\n", model_input->dims->size);
  // logger.info("model_input->dims->data[0]: %d\n", model_input->dims->data[0]);

  // logger.info("kFeatureSliceSize: %d\n", model_input->dims->data[1]);

  // // logger.info("kFeatureSliceCount: %d\n", model_input->dims->data[2]);

  // logger.info("model_input->type: %d\n", model_input->type);
  // Get information about the memory area to use for the model's input.
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceSize) ||
      // (model_input->dims->data[2] != kFeatureSliceCount) ||
      (model_input->type != kTfLiteFloat32))
  {
    logger.info("Bad input tensor parameters in model\n");

    // TF_LITE_REPORT_ERROR(error_reporter,
    //                      "Bad input tensor parameters in model");
    return;
  }
  // logger.debug("MODEL OK\n");
  float in_x = 0;
  for (int j = 0; j < 1; j++)
  {
    model_input_buffer = model_input->data.uint8;
    for (int i = 0; i < kFeatureBufferSize; i++)
    {
      feature_buffer[i] = static_cast<float>(in_x);
    }
    // Copy feature buffer to input tensor
    memcpy(model_input_buffer, feature_buffer,
           kFeatureBufferSize * sizeof(float));
    // logger.printf("TFLite invoked\n");
    t.reset();
    t.start();
    // mbed_mem_trace_set_callback(mem_trace_callback);

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk)
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }
    t.stop();

    // Obtain a pointer to the output tensor
    TfLiteTensor *output = interpreter->output(0);

    ThisThread::sleep_for(1000);

    logger.info("Bytes allocated: %ld\n", bytes_allocated);
    logger.info("Finished %dus\r", t.read_us());

    ThisThread::sleep_for(1000);

    // logger.printf("Finished %dus\r", t.read_us());
    // logger.printf("output->type:%u \n", output->type);
    // logger.printf("output->name:%s \n", output->name);
    // logger.printf("output->bytes:%u \n", output->bytes);

    // logger.info("\nFinished %dus\n", t.read_us());
    in_x++;
    while (true)
    {
    }
  }
  while (!_is_stopped)
  {
    t.reset();
    t.start();
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk)
    {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }
    t.stop();

    // Obtain a pointer to the output tensor
    TfLiteTensor *output = interpreter->output(0);
    logger.printf("TFLite Finished %dus\r", t.read_us());
    // logger.printf("output->type:%u \n", output->type);
    // logger.printf("output->name:%s \n", output->name);
    // logger.printf("output->bytes:%u \n", output->bytes);
    ThisThread::sleep_for(10);
  }
  while (!_is_stopped)
  {
    osEvent evt = _mail_box_rx.get();
    if (evt.status == osEventMail)
    {
      mail_t *mail = static_cast<mail_t *>(evt.value.p);
      switch (mail->type)
      {
      case MessageType_t::NEW_AUDIO_FRAME_EVENT:
      {
        // _logger.printf("NEW_AUDIO_FRAME_EVENT frame.size: %u\n",
        // mail->frame.size());
        static uint32_t offset = 0;
        constexpr auto RECOGNIZE_THRESHOLD = 6000000;
        constexpr auto MAX_RMS = 200000000;
        constexpr auto NUM_OF_DIVISIONS = 30;
        size_t words_left = kPhraseBufferSize - offset - 1;
        size_t words_n =
            mail->frame.size() < words_left ? mail->frame.size() : words_left;
        uint32_t wave_rms = 0;
        for (int i = 0; i < words_n; i++)
        {
          // mail->frame[i] *= 4;
          _phrase_buf[offset + i] = mail->frame[i];
          // _phrase_buf->push_back(static_cast<int16_t>(mail->frame[i]));
          wave_rms += mail->frame[i] * mail->frame[i];
        }
        wave_rms /= words_n;
      std:
        string tmp_str("\rwave_rms:");
        bool thresh_flag = false;
        for (int i = 0; i < NUM_OF_DIVISIONS; i++)
        {
          if (log(static_cast<double>(i * (MAX_RMS / NUM_OF_DIVISIONS))) >=
                  log(static_cast<double>(RECOGNIZE_THRESHOLD)) &&
              !thresh_flag)
          {
            thresh_flag = true;
            tmp_str += "]";
          }
          else if (log(static_cast<double>(i *
                                           (MAX_RMS / NUM_OF_DIVISIONS))) <=
                   log(static_cast<double>(wave_rms)))
          {
            tmp_str += "|";
          }
          else
          {
            tmp_str += ".";
          }
        }
        // logger.printf("%s", tmp_str.c_str());

        mail->frame.clear();
        offset += words_n;
        if (wave_rms > RECOGNIZE_THRESHOLD && !is_recording_started)
        {
          is_recording_started = true;
        }
        else if (wave_rms <= RECOGNIZE_THRESHOLD && !is_recording_started &&
                 offset > 2 * kPhraseBufferSize / 3)
        {
          offset = 0;
        }
        // logger.printf("offset: %u\n", offset);
        if (offset >= kPhraseBufferSize - 1)
        {
          _is_ready = false;
          logger.debug("\n");
          logger.debug("MFCC Started\n");
          t.reset();
          t.start();
          offset = 0;
          for (int i = 0; i <= 2 * kPhraseBufferSize - 2 * WINDOW_SIZE;
               i += 2 * HOP_SIZE)
          {
            //   int16_t *frame_buffer_tmp = new int16_t[WINDOW_SIZE];
            memcpy(tmpbuf, reinterpret_cast<uint8_t *>(_phrase_buf) + i,
                   WINDOW_SIZE * 2);
            //   memcpy(tmpbuf, _phrase_buf, WINDOW_SIZE * 2);
            //   for(int j = 0; j < WINDOW_SIZE; j++)
            //   {
            //       frame_buffer_tmp[j] = _phrase_buf->at(j);
            //   }
            //   _phrase_buf->erase(_phrase_buf->begin(),_phrase_buf->begin()+WINDOW_SIZE);
            _mfcc->processFrame<v_f_t>(tmpbuf, WINDOW_SIZE, _feature_vector);
            //   std::string outstr = _mfcc->processFrame(tmpbuf,
            //   WINDOW_SIZE); logger.printf("%s\n", outstr.c_str());
            //   delete[] frame_buffer_tmp;
          }
          // logger.printf(
          //     "MFCC Finished %2.2fs _feature_vector.size() : %u
          //     WINDOW_SIZE: "
          //     "%u kPhraseBufferSize: %u\n",
          //     t.read(), _feature_vector.size(), WINDOW_SIZE,
          //     kPhraseBufferSize);

          constexpr auto kScanStep =
              1 * kFeatureBufferSize / kFeatureSliceSize;
          unsigned int counter_scan[kHotwordsNumber] = {0, 0, 0, 0,
                                                        0, 0, 0, 0};
          std::vector<unsigned int> scan_results;
          for (int scan_offset = 0;
               scan_offset <= _feature_vector.size() - kFeatureBufferSize;
               scan_offset += kScanStep)
          {
            for (int i = 0; i < kFeatureBufferSize; i++)
            {
              feature_buffer[i] = _feature_vector[i + scan_offset];
            }
            // Copy feature buffer to input tensor
            memcpy(model_input_buffer, feature_buffer,
                   kFeatureBufferSize * sizeof(float));
            // logger.printf("TFLite invoked\n");
            t.reset();
            t.start();
            TfLiteStatus invoke_status = interpreter->Invoke();

            if (invoke_status != kTfLiteOk)
            {
              TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
              return;
            }
            t.stop();

            // Obtain a pointer to the output tensor
            TfLiteTensor *output = interpreter->output(0);
            // logger.printf("TFLite Finished %2.2fs\n", t.read());
            // logger.printf("output->type:%u \n", output->type);
            // logger.printf("output->name:%s \n", output->name);
            // logger.printf("output->bytes:%u \n", output->bytes);
            int max_val_idx = 0;
            static int max_val_idx_prev = 0;

            float max_val = 0;
            for (int i = 0; i < output->bytes; i += sizeof(float))
            {
              float value =
                  *reinterpret_cast<float *>(output->data.uint8 + i);
              if (value > max_val)
              {
                max_val = value;
                max_val_idx = i / sizeof(float);
              }
              logger.debug("%s:%u ", kHotwordsDict[i / sizeof(float)],
                           static_cast<unsigned int>(value * 100));
            }
            scan_results.push_back(max_val_idx);
            //"могу", "буду", "хочу", "стоп", "да", "нет", "noise",
            //"unknown"
            if ((max_val_idx == 4) || (max_val_idx == 7))
            {
              send_listen_event(Teddy::ListenNode::Hotword::Hotword_EnYes);
              logger.debug("send_listen_event Hotword_EnYes\n");
            }
            else if ((max_val_idx == 5))
            {
              send_listen_event(Teddy::ListenNode::Hotword::Hotword_EnNo);
              logger.debug("send_listen_event Hotword_EnYes\n");
            }
            max_val_idx_prev = max_val_idx;
            // logger.printf("t: %ums\n%s\n", t.read_ms(),
            //               kHotwordsDict[max_val_idx]);
          }

          logger.debug("\n");
          _feature_vector.clear();
          _is_ready = true;
          is_recording_started = false;
        }
        break;
      }
      default:
      {
        break;
      }
      }
      while (_mail_box_rx.free(mail) != osOK)
      {
        ThisThread::sleep_for(10);
      }
    }
    // ThisThread::sleep_for(1);
  }
}