#ifndef MEMRYX_BACKEND_HPP_
#define MEMRYX_BACKEND_HPP_

#include "inference_backend_base.hpp"

#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <memx/accl/MxAccl.h>
#include <memx/accl/utils/featureMap.h>
#include <memx/accl/utils/mxTypes.h>

namespace autoware_pov::vision
{

// MemryX MX3 backend. Loads a .dfp Dataflow Program produced by the
// NeuralCompiler and exposes a synchronous doInference interface by wrapping
// the underlying async stream/callback runtime (MxAccl). The mxa-manager
// daemon must be running for shared-mode operation (the default).
class MemryXBackend : public InferenceBackend
{
public:
  // The precision and gpu_id arguments are accepted for parity with the
  // OnnxRuntime / TensorRT backends but are ignored — DFP precision is baked
  // in at compile time and the MX3 is the only device.
  MemryXBackend(const std::string & dfp_path, const std::string & precision, int gpu_id);
  ~MemryXBackend() override;

  bool doInference(const cv::Mat & input_image) override;

  const float * getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;

  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  void preprocess(const cv::Mat & image, std::vector<float> & buffer);
  bool inputCallback(std::vector<const MX::Types::FeatureMap *> in_fmaps, int stream_id);
  bool outputCallback(std::vector<const MX::Types::FeatureMap *> out_fmaps, int stream_id);

  std::unique_ptr<MX::Runtime::MxAccl> accl_;

  int model_input_height_;
  int model_input_width_;
  std::vector<int64_t> output_shape_;  // NCHW per the original model shape
  size_t input_elements_;              // 1 * 3 * H * W
  size_t output_elements_;             // product of output_shape_

  // Async-to-sync bridge.  doInference stages preprocessed input, then waits
  // on the output condition variable; the runtime's worker threads drive the
  // two callbacks below.
  std::mutex io_mutex_;
  std::condition_variable cv_input_ready_;
  std::condition_variable cv_output_ready_;
  bool input_ready_ = false;
  bool output_ready_ = false;
  std::atomic<bool> shutting_down_{false};

  std::vector<float> input_buffer_;
  std::vector<float> output_buffer_;
};

}  // namespace autoware_pov::vision

#endif  // MEMRYX_BACKEND_HPP_
