#include "../include/hailo_backend.hpp"

#include "../include/logging.hpp"

#include <cstring>
#include <stdexcept>

namespace autoware_pov::vision
{

namespace
{

// Tiny helper: hailort APIs return Expected<T>; throw on error so the backend
// constructor's failure surfaces cleanly through the run_model_node factory.
template <typename T>
T expect_or_throw(hailort::Expected<T> && exp, const std::string & what)
{
  if (!exp) {
    throw std::runtime_error(
      "HailoBackend: " + what + " failed (status=" +
      std::to_string(static_cast<int>(exp.status())) + ")");
  }
  return exp.release();
}

}  // namespace

HailoBackend::HailoBackend(const std::string & hef_path,
                           const std::string & /*precision*/, int /*gpu_id*/)
{
  LOG_INFO("Hailo: loading HEF %s", hef_path.c_str());

  // VDevice with ROUND_ROBIN scheduler (skill Rule 2 — HailoRT manages
  // activation per infer() call). Single device on this host; the auto
  // selection is fine. To pin to a specific BDF, populate device_ids.
  hailo_vdevice_params_t vdevice_params{};
  auto default_status = hailo_init_vdevice_params(&vdevice_params);
  if (default_status != HAILO_SUCCESS) {
    throw std::runtime_error(
      "HailoBackend: hailo_init_vdevice_params failed (" +
      std::to_string(static_cast<int>(default_status)) + ")");
  }
  vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
  vdevice_ = expect_or_throw(
    hailort::VDevice::create(vdevice_params), "VDevice::create");

  // Configure the HEF onto the device.
  auto hef = expect_or_throw(hailort::Hef::create(hef_path), "Hef::create");

  auto configure_params = expect_or_throw(
    hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE),
    "Hef::create_configure_params");

  auto network_groups = expect_or_throw(
    vdevice_->configure(hef, configure_params), "VDevice::configure");
  if (network_groups.size() != 1) {
    throw std::runtime_error(
      "HailoBackend: expected exactly 1 network group in HEF, got " +
      std::to_string(network_groups.size()));
  }
  network_group_ = network_groups[0];

  // Probe input/output vstream shapes BEFORE creating the pipeline so we know
  // buffer sizes ahead of time.
  auto in_infos = expect_or_throw(
    network_group_->get_input_vstream_infos(), "get_input_vstream_infos");
  if (in_infos.size() != 1) {
    throw std::runtime_error(
      "HailoBackend: only single-input HEFs supported (got " +
      std::to_string(in_infos.size()) + ")");
  }
  auto out_infos = expect_or_throw(
    network_group_->get_output_vstream_infos(), "get_output_vstream_infos");
  if (out_infos.size() != 1) {
    LOG_WARN("Hailo: HEF has %zu output vstreams; backend exposes only the first",
             out_infos.size());
  }

  input_vstream_name_ = in_infos[0].name;
  output_vstream_name_ = out_infos[0].name;

  // Input shape is reported as (height, width, features) where features = C.
  model_input_height_ = static_cast<int>(in_infos[0].shape.height);
  model_input_width_ = static_cast<int>(in_infos[0].shape.width);
  model_input_channels_ = static_cast<int>(in_infos[0].shape.features);
  input_bytes_ = static_cast<size_t>(model_input_height_) *
                 model_input_width_ * model_input_channels_;
  input_buffer_.resize(input_bytes_);

  // Output is float NCHW [1, C, H, W]. We override the user_buffer_format to
  // NCHW so HailoRT does the transpose + dequant inside its worker threads —
  // matches what run_model_node expects from the OnnxRuntime/TensorRT backends.
  const auto out_h = static_cast<int64_t>(out_infos[0].shape.height);
  const auto out_w = static_cast<int64_t>(out_infos[0].shape.width);
  const auto out_c = static_cast<int64_t>(out_infos[0].shape.features);
  output_shape_nchw_ = {1, out_c, out_h, out_w};
  output_floats_ = static_cast<size_t>(out_c * out_h * out_w);
  output_buffer_.resize(output_floats_);

  // Build vstream params. Input keeps the HEF's native NHWC uint8 layout
  // (so the host preprocessing reduces to resize + BGR2RGB). Output is
  // FLOAT32 + NCHW so HailoRT dequantizes and transposes for us.
  auto input_params = expect_or_throw(
    network_group_->make_input_vstream_params(
      false, HAILO_FORMAT_TYPE_UINT8,
      HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
      HAILO_DEFAULT_VSTREAM_QUEUE_SIZE),
    "make_input_vstream_params");
  auto output_params = expect_or_throw(
    network_group_->make_output_vstream_params(
      false, HAILO_FORMAT_TYPE_FLOAT32,
      HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
      HAILO_DEFAULT_VSTREAM_QUEUE_SIZE),
    "make_output_vstream_params");

  // Override the output's spatial order to NCHW. Default after FormatType is
  // typically NHWC; we want NCHW so run_model_node's class-iteration loop
  // walks contiguous channel-planes.
  for (auto & kv : output_params) {
    kv.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NCHW;
  }

  infer_pipeline_ = std::make_unique<hailort::InferVStreams>(expect_or_throw(
    hailort::InferVStreams::create(*network_group_, input_params, output_params),
    "InferVStreams::create"));

  LOG_INFO("Hailo: input  uint8 NHWC %dx%dx%d (%zu bytes)  vstream='%s'",
           model_input_height_, model_input_width_, model_input_channels_,
           input_bytes_, input_vstream_name_.c_str());
  LOG_INFO("Hailo: output float NCHW [1, %ld, %ld, %ld] (%zu floats)  vstream='%s'",
           static_cast<long>(out_c), static_cast<long>(out_h),
           static_cast<long>(out_w), output_floats_,
           output_vstream_name_.c_str());
}

HailoBackend::~HailoBackend()
{
  // Explicit teardown order so worker threads + DMA buffers drain before
  // the VDevice closes (skill Rule 2 best practice).
  infer_pipeline_.reset();
  network_group_.reset();
  vdevice_.reset();
}

bool HailoBackend::doInference(const cv::Mat & input_image)
{
  // Host preprocessing: resize + BGR2RGB into the persistent uint8 NHWC
  // buffer. /255 + ImageNet normalize were folded into the input layer's
  // quant scales at compile time (Hailo's `normalization` model script
  // command), so the chip sees the right floats once the input wrapper
  // applies them.
  cv::Mat resized;
  cv::resize(input_image, resized, cv::Size(model_input_width_, model_input_height_));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  if (!rgb.isContinuous()) {
    rgb = rgb.clone();
  }
  if (rgb.total() * rgb.elemSize() != input_bytes_) {
    LOG_ERROR("Hailo: preprocessed buffer size mismatch (got %zu, expected %zu)",
              rgb.total() * rgb.elemSize(), input_bytes_);
    return false;
  }
  std::memcpy(input_buffer_.data(), rgb.data, input_bytes_);

  std::map<std::string, hailort::MemoryView> input_view{
    {input_vstream_name_,
     hailort::MemoryView(input_buffer_.data(), input_bytes_)}};
  std::map<std::string, hailort::MemoryView> output_view{
    {output_vstream_name_,
     hailort::MemoryView(output_buffer_.data(),
                         output_floats_ * sizeof(float))}};

  const auto status = infer_pipeline_->infer(input_view, output_view, 1);
  if (status != HAILO_SUCCESS) {
    LOG_ERROR("Hailo: InferVStreams::infer failed (status=%d)",
              static_cast<int>(status));
    return false;
  }
  return true;
}

const float * HailoBackend::getRawTensorData() const
{
  return output_buffer_.data();
}

std::vector<int64_t> HailoBackend::getTensorShape() const
{
  return output_shape_nchw_;
}

}  // namespace autoware_pov::vision
