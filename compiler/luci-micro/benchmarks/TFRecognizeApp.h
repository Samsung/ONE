#ifndef __TFRECOGNIZEAPP_H
#define __TFRECOGNIZEAPP_H
#include <vector>

#include "ICore.h"
#include "ILogger.h"
#include "IRecognizeCore.h"
#include "MFCC.h"
#include "mbed.h"
#include "rtos.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
namespace Teddy
{
    namespace AI
    {
        // Globals, used for compatibility with Arduino-style sketches.
        namespace
        {
            typedef std::vector<float> v_f_t;
            tflite::ErrorReporter *error_reporter = nullptr;
            const tflite::Model *model = nullptr;
            tflite::MicroInterpreter *interpreter = nullptr;
            TfLiteTensor *model_input = nullptr;
            int32_t previous_time = 0;

            // Create an area of memory to use for input, output, and intermediate arrays.
            // The size of this will depend on the model you're using, and may need to be
            // determined by experimentation.
            constexpr auto kHotwordsNumber = 8;
            const char *kHotwordsDict[kHotwordsNumber] = {
                "могу", "буду", "хочу", "стоп", "да", "нет", "noise", "unknown"};
            constexpr auto kTensorArenaSize = 200000;
            constexpr auto kFeatureSliceSize = 1;
            constexpr auto kFeatureSliceCount = 1;
            constexpr auto kMFCCNumberOfFilters = 12;
            constexpr auto kMFCCLowestFrequency = 50;
            constexpr auto kMFCCHighestFrequency = 2666;
            constexpr auto kMFCCSamplerate = 16000;
            constexpr auto kMFCCNumberOfCepstral = kFeatureSliceSize;
            constexpr auto kMFCCWinLength = 96; // 48 96 182
            constexpr auto kMFCCFrameShift = 96;
            constexpr unsigned int WINDOW_SIZE = static_cast<unsigned int>(2 * 512);
            constexpr unsigned int HOP_SIZE = static_cast<unsigned int>(2 * 512);

            constexpr auto kPhraseBufferSec = 0.5;
            constexpr size_t kPhraseBufferSize =
                256 * 4 * 8 * 1 +
                1; // static_cast<size_t>(kMFCCSamplerate * kPhraseBufferSec + 0.5) + 1;
            constexpr auto kFeatureBufferSize = kFeatureSliceCount * kFeatureSliceSize;
            // uint8_t feature_buffer[kFeatureElementCount];
            uint8_t *model_input_buffer = nullptr;
        }; // namespace
        class TFRecognizeApp : public IRecognizeCore, public ICore
        {
        public:
            TFRecognizeApp(const osPriority_t thread_priority, const char *name,
                           Mail<mail_t, 8> &mail_box_rx, QueueMMPC_t &mail_box_tx,
                           const tflite::Model *model, ILogger &logger,
                           uint8_t *tensor_arena = nullptr,
                           uint8_t *recording_buffer = nullptr);
            void process_frame(std::vector<int16_t> vec) override;
            virtual ~TFRecognizeApp(void);

        private:
            Mail<mail_t, 8> &_mail_box_rx;
            QueueMMPC_t &_mail_box_tx;
            const tflite::Model *_model;
            uint8_t *_tensor_arena;
            Math::MFCC *_mfcc = nullptr;
            ILogger &logger;
            v_f_t _feature_vector;
            bool _is_ready = false;
            // std::vector<int16_t> *_phrase_buf;
            int16_t *_phrase_buf;

            void send_listen_event(Teddy::ListenNode::Hotword hotword);

        protected:
            void setup(void) override;
            void loop(void) override;
        };
    }; // namespace AI
};     // namespace Teddy
#endif //__RECOGNIZEAPP_H