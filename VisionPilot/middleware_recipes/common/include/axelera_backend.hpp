#ifndef AXELERA_BACKEND_HPP_
#define AXELERA_BACKEND_HPP_

#include "inference_backend_base.hpp"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <axruntime/axruntime.h>

namespace autoware_pov::vision
{

// Axelera Metis backend. Loads a compiled-model directory produced by
// `axcompile` (manifest.json + model_*.json + kernel_function.c + pool_*.bin)
// and exposes a synchronous doInference interface by wrapping the
// axruntime C API.
//
// Per the mb-axelera skill (Rules 1-5):
//   * Each model claims `num_sub_devices` based on its L2 constant size
//     (default 1 core for a single Lite-family model; SceneSegLite+Scene3DLite
//     can co-load on one Metis M.2).
//   * Per-tensor quant params (scale, zero_point) and padding spec come from
//     `axrTensorInfo` directly - no need to parse manifest.json separately.
//   * Padding is filled with the quantization zero-point, NOT zero. Padding
//     a quantized tensor with float-zero produces wrong-tensor noise that
//     the model will happily run on.
//   * Outputs are returned in the model's declared order, not by name.
class AxeleraBackend : public InferenceBackend
{
public:
  // precision/gpu_id accepted for parity with the OnnxRuntime / TensorRT
  // backends but ignored — INT8 precision is baked into the compiled
  // artifact at compile time and the Metis is the only device.
  AxeleraBackend(const std::string & model_dir, const std::string & precision, int gpu_id);
  ~AxeleraBackend() override;

  bool doInference(const cv::Mat & input_image) override;

  const float * getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;

  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  // RAII handle that calls axr_destroy on scope exit.
  template <typename T>
  struct AxrDeleter {
    void operator()(T * p) const noexcept;
  };
  template <typename T>
  using AxrPtr = std::unique_ptr<T, AxrDeleter<T>>;

  // Resource-handles are non-owning pointers managed by the context (devices
  // come from axr_list_devices and are not destroyed by the caller).
  AxrPtr<axrContext> context_;
  AxrPtr<axrConnection> connection_;
  AxrPtr<axrModel> model_;
  AxrPtr<axrModelInstance> instance_;

  axrTensorInfo input_info_{};
  axrTensorInfo output_info_{};

  int model_input_height_;
  int model_input_width_;
  int model_input_channels_;

  // Padded-shape NHWC buffer (the wire format the runtime expects), filled
  // with the quantization zero-point before each frame's data is written into
  // the unpadded sub-region.
  std::vector<uint8_t> input_padded_;
  size_t input_padded_bytes_;
  uint8_t input_zero_point_;

  // Padded int8 output buffer + the dequantized float NCHW user buffer.
  std::vector<int8_t> output_padded_;
  size_t output_padded_bytes_;
  std::vector<float> output_dequant_;
  std::vector<int64_t> output_shape_;  // NCHW, unpadded
};

}  // namespace autoware_pov::vision

#endif  // AXELERA_BACKEND_HPP_
