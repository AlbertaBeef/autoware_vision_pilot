#ifndef HAILO_BACKEND_HPP_
#define HAILO_BACKEND_HPP_

#include "inference_backend_base.hpp"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>
#include <hailo/inference_pipeline.hpp>

namespace autoware_pov::vision
{

// Hailo-8 backend. Loads a .hef produced by `hailo parser/optimize/compile`
// (see Models/exports/hailo/hailo_flow.py) and exposes a synchronous
// doInference interface via HailoRT's InferVStreams pipeline.
//
// Design notes:
// - Persistent VDevice + ConfiguredNetworkGroup + InferVStreams across the
//   lifetime of the backend (skill rule 2: never tear them down per call).
// - ROUND_ROBIN scheduler enabled — HailoRT handles activation per infer call.
// - Output vstreams configured FLOAT32 + NCHW so doInference can hand back a
//   float* directly compatible with run_model_node's downstream argmax. The
//   ImageNet mean/std + /255 normalize was folded into the input layer at
//   compile time, so the host feeds raw uint8 NHWC at the input layer shape.
class HailoBackend : public InferenceBackend
{
public:
  // precision / gpu_id are accepted for parity with the OnnxRuntime / TensorRT
  // backends but ignored — INT8 precision is baked into the .hef at compile
  // time and the Hailo accelerator is selected by PCIe BDF (not GPU id).
  HailoBackend(const std::string & hef_path, const std::string & precision, int gpu_id);
  ~HailoBackend() override;

  bool doInference(const cv::Mat & input_image) override;

  const float * getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;

  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  // Order matters: vdevice must outlive network_group must outlive
  // infer_pipeline. Destruction runs bottom-up.
  std::unique_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
  std::unique_ptr<hailort::InferVStreams> infer_pipeline_;

  std::string input_vstream_name_;
  std::string output_vstream_name_;

  int model_input_height_;
  int model_input_width_;
  int model_input_channels_;
  size_t input_bytes_;

  // Output shape in NCHW (the order we request from HailoRT). Stored as
  // [1, C, H, W] int64 so getTensorShape() matches the ONNX/TensorRT
  // convention the rest of run_model_node consumes.
  std::vector<int64_t> output_shape_nchw_;
  size_t output_floats_;

  // Reusable scratch buffers (avoid per-frame allocation).
  std::vector<uint8_t> input_buffer_;
  std::vector<float> output_buffer_;
};

}  // namespace autoware_pov::vision

#endif  // HAILO_BACKEND_HPP_
