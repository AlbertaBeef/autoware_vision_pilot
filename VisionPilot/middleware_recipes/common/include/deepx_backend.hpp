#ifndef DEEPX_BACKEND_HPP_
#define DEEPX_BACKEND_HPP_

#include "inference_backend_base.hpp"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <dxrt/dxrt_api.h>

namespace autoware_pov::vision
{

// DeepX M1 backend. Loads a .dxnn produced by dxcom and exposes a synchronous
// doInference interface via dxrt::InferenceEngine::Run.
//
// DX-COM bakes Div(/255) + Normalize(ImageNet) into the .dxnn input wrapper,
// silently rewrites NCHW source to NHWC at the boundary, and bakes both
// transpose(HWC->NCHW) and expandDim(batch) into the same wrapper. The host
// therefore feeds a raw uint8 NHWC buffer; the device returns float NCHW.
class DeepXBackend : public InferenceBackend
{
public:
  // precision/gpu_id accepted for parity with the OnnxRuntime / TensorRT
  // backends but ignored — INT8 precision is baked into the .dxnn at compile
  // time and the M1 is the only device.
  DeepXBackend(const std::string & dxnn_path, const std::string & precision, int gpu_id);
  ~DeepXBackend() override = default;

  bool doInference(const cv::Mat & input_image) override;

  const float * getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;

  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  std::unique_ptr<dxrt::InferenceEngine> engine_;

  int model_input_height_;
  int model_input_width_;
  int model_input_channels_;
  size_t input_bytes_;

  // Cached after each Run().  TensorPtrs owns the device-allocated buffers
  // until the next Run() invocation, so we hold the shared_ptrs explicitly.
  dxrt::TensorPtrs last_outputs_;
  std::vector<int64_t> output_shape_;

  // Reusable preprocessing scratch buffer (avoids per-frame allocation).
  std::vector<uint8_t> input_buffer_;
};

}  // namespace autoware_pov::vision

#endif  // DEEPX_BACKEND_HPP_
