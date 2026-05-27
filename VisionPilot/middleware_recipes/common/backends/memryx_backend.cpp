#include "../include/memryx_backend.hpp"

#include "../include/logging.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

namespace autoware_pov::vision
{

namespace
{
constexpr int kStreamId = 0;
constexpr int kModelId = 0;
}  // namespace

MemryXBackend::MemryXBackend(const std::string & dfp_path,
                             const std::string & /*precision*/, int /*gpu_id*/)
{
  LOG_INFO("MemryX: loading DFP %s", dfp_path.c_str());

  // Default arguments: device 0, preserve original model shape on both ends
  // (NCHW), shared mode via mxa-manager UNIX socket.
  accl_ = std::make_unique<MX::Runtime::MxAccl>(std::filesystem::path(dfp_path));

  const auto info = accl_->get_model_info(kModelId);

  if (info.num_in_featuremaps != 1 || info.num_out_featuremaps != 1) {
    throw std::runtime_error(
      "MemryXBackend: only single-input / single-output models are supported "
      "(model has " + std::to_string(info.num_in_featuremaps) + " inputs, " +
      std::to_string(info.num_out_featuremaps) + " outputs)");
  }

  const auto & in_shape = info.in_raw_shapes.at(0);   // NCHW from PyTorch export
  const auto & out_shape = info.out_raw_shapes.at(0);
  if (in_shape.size() != 4) {
    throw std::runtime_error("MemryXBackend: expected 4D NCHW input shape");
  }
  // NCHW: [N, C, H, W]
  model_input_height_ = static_cast<int>(in_shape[2]);
  model_input_width_ = static_cast<int>(in_shape[3]);
  output_shape_.assign(out_shape.begin(), out_shape.end());

  input_elements_ = static_cast<size_t>(
    std::accumulate(in_shape.begin(), in_shape.end(), int64_t{1}, std::multiplies<int64_t>()));
  output_elements_ = static_cast<size_t>(
    std::accumulate(out_shape.begin(), out_shape.end(), int64_t{1}, std::multiplies<int64_t>()));

  input_buffer_.resize(input_elements_);
  output_buffer_.resize(output_elements_);

  const auto shape_to_str = [](const std::vector<int64_t> & shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) s += ",";
      s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
  };
  const std::string in_shape_str = shape_to_str(in_shape);
  const std::string out_shape_str = shape_to_str(out_shape);
  LOG_INFO("MemryX: input %s, output %s", in_shape_str.c_str(), out_shape_str.c_str());

  using namespace std::placeholders;
  MX::Types::callback_t in_cb = std::bind(&MemryXBackend::inputCallback, this, _1, _2);
  MX::Types::callback_t out_cb = std::bind(&MemryXBackend::outputCallback, this, _1, _2);
  accl_->connect_stream(in_cb, out_cb, kStreamId, kModelId);
  accl_->start(kModelId);
}

MemryXBackend::~MemryXBackend()
{
  {
    std::lock_guard<std::mutex> lk(io_mutex_);
    shutting_down_ = true;
    input_ready_ = true;  // wake the input callback so it can return false
  }
  cv_input_ready_.notify_all();

  if (accl_) {
    try {
      accl_->wait(kModelId);
    } catch (const std::exception & e) {
      LOG_WARN("MemryX: wait() during shutdown threw: %s", e.what());
    }
  }
}

void MemryXBackend::preprocess(const cv::Mat & image, std::vector<float> & buffer)
{
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(model_input_width_, model_input_height_));
  cv::Mat floatm;
  resized.convertTo(floatm, CV_32FC3, 1.0 / 255.0);

  // Match the existing OnnxRuntime / TensorRT backends: BGR input from
  // cv_bridge, normalize using BGR-ordered ImageNet statistics so the values
  // delivered per-channel match what the model was trained against.
  cv::subtract(floatm, cv::Scalar(0.406, 0.456, 0.485), floatm);
  cv::divide(floatm, cv::Scalar(0.225, 0.224, 0.229), floatm);

  const size_t plane = static_cast<size_t>(model_input_height_) * model_input_width_;
  buffer.resize(plane * 3);
  std::vector<cv::Mat> channels(3);
  cv::split(floatm, channels);
  // HWC -> CHW (NCHW with N=1)
  std::memcpy(buffer.data() + 0 * plane, channels[0].data, plane * sizeof(float));
  std::memcpy(buffer.data() + 1 * plane, channels[1].data, plane * sizeof(float));
  std::memcpy(buffer.data() + 2 * plane, channels[2].data, plane * sizeof(float));
}

bool MemryXBackend::doInference(const cv::Mat & input_image)
{
  preprocess(input_image, input_buffer_);

  {
    std::lock_guard<std::mutex> lk(io_mutex_);
    input_ready_ = true;
    output_ready_ = false;
  }
  cv_input_ready_.notify_one();

  std::unique_lock<std::mutex> lk(io_mutex_);
  const bool ok = cv_output_ready_.wait_for(
    lk, std::chrono::seconds(5),
    [this] { return output_ready_ || shutting_down_.load(); });
  if (!ok) {
    LOG_ERROR("MemryX: inference timed out waiting for output");
    return false;
  }
  return !shutting_down_.load();
}

const float * MemryXBackend::getRawTensorData() const
{
  if (output_buffer_.empty()) {
    throw std::runtime_error("MemryXBackend: getRawTensorData() before doInference()");
  }
  return output_buffer_.data();
}

std::vector<int64_t> MemryXBackend::getTensorShape() const
{
  return output_shape_;
}

bool MemryXBackend::inputCallback(std::vector<const MX::Types::FeatureMap *> in_fmaps,
                                  int /*stream_id*/)
{
  std::unique_lock<std::mutex> lk(io_mutex_);
  cv_input_ready_.wait(lk, [this] { return input_ready_ || shutting_down_.load(); });
  if (shutting_down_.load()) {
    return false;  // signals the runtime to end the stream
  }
  in_fmaps.at(0)->set_data(input_buffer_.data());
  input_ready_ = false;
  return true;
}

bool MemryXBackend::outputCallback(std::vector<const MX::Types::FeatureMap *> out_fmaps,
                                   int /*stream_id*/)
{
  out_fmaps.at(0)->get_data(output_buffer_.data());
  {
    std::lock_guard<std::mutex> lk(io_mutex_);
    output_ready_ = true;
  }
  cv_output_ready_.notify_one();
  return true;
}

}  // namespace autoware_pov::vision
