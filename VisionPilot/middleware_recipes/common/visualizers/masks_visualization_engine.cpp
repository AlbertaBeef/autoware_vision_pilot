#include "../include/masks_visualization_engine.hpp"

namespace autoware_pov::common {

MasksVisualizationEngine::MasksVisualizationEngine(const std::string& viz_type, bool show_opencv_window) 
    : viz_type_(viz_type), show_opencv_window_(show_opencv_window), 
      opencv_frame_count_(0), opencv_fps_(0.0) {
    last_opencv_time_ = std::chrono::steady_clock::now();
}

cv::Mat MasksVisualizationEngine::visualize(const cv::Mat& mask, const cv::Mat& original_image, 
                                           double latency_ms, double fps) {
    auto colorize_start_time = std::chrono::steady_clock::now();
    
    cv::Mat color_mask = createColorMask(mask);
    
    auto colorize_end_time = std::chrono::steady_clock::now();
    auto colorize_time_ms = std::chrono::duration<double, std::milli>(colorize_end_time - colorize_start_time).count();
    
    // Resize color mask to match original image if needed
    cv::Mat resized_color_mask;
    if (color_mask.size() != original_image.size()) {
        cv::resize(color_mask, resized_color_mask, original_image.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        resized_color_mask = color_mask;
    }
    
    // Blend with original image: 50% color + 50% original
    cv::Mat blended_result;
    cv::addWeighted(resized_color_mask, 0.5, original_image, 0.5, 0.0, blended_result);
    
    // Display OpenCV window if enabled
    if (show_opencv_window_) {
        displayOpenCVWindow(blended_result, latency_ms, fps, colorize_time_ms);
    }
    
    return blended_result;
}

cv::Mat MasksVisualizationEngine::createColorMask(const cv::Mat& mask) {
    cv::Mat color_mask = cv::Mat::zeros(mask.size(), CV_8UC3);

    if (viz_type_ == "scene") {
        // Scene segmentation: binary masks (0/255) → red foreground
        cv::Mat red_mask;
        cv::inRange(mask, cv::Scalar(1), cv::Scalar(255), red_mask);
        color_mask.setTo(cv::Scalar(0, 0, 255), red_mask);  // BGR: Red
    } else if (viz_type_ == "domain") {
        color_mask.setTo(cv::Scalar(255, 93, 61),  mask == 0);     // BGR: Orange
        color_mask.setTo(cv::Scalar(145, 28, 255), mask == 255);   // BGR: Purple
    } else if (viz_type_ == "egolanes") {
        color_mask.setTo(cv::Scalar(255, 0, 0),   mask == 0);     // ego-left
        color_mask.setTo(cv::Scalar(255, 0, 200),  mask == 1);     // ego-right
        color_mask.setTo(cv::Scalar(0, 153, 0),    mask == 2);     // all_other_lanes
    } else if (viz_type_ == "scene_seg_lite") {
        // Cityscapes 19-class palette (BGR ordered for OpenCV).
        // Class IDs match SCENESEGLITE_DEFAULT_CONFIG["loss"]["class_names"].
        static const cv::Vec3b kCityscapesBgr[19] = {
            {128,  64, 128},  //  0 road
            {232,  35, 244},  //  1 sidewalk
            { 70,  70,  70},  //  2 building
            {156, 102, 102},  //  3 wall
            {153, 153, 190},  //  4 fence
            {153, 153, 153},  //  5 pole
            { 30, 170, 250},  //  6 traffic_light
            {  0, 220, 220},  //  7 traffic_sign
            { 35, 142, 107},  //  8 vegetation
            {152, 251, 152},  //  9 terrain
            {180, 130,  70},  // 10 sky
            { 60,  20, 220},  // 11 person
            {  0,   0, 255},  // 12 rider
            {142,   0,   0},  // 13 car
            { 70,   0,   0},  // 14 truck
            {100,  60,   0},  // 15 bus
            {100,  80,   0},  // 16 train
            {230,   0,   0},  // 17 motorcycle
            { 32,  11, 119},  // 18 bicycle
        };
        for (int y = 0; y < mask.rows; ++y) {
            const uint8_t* mrow = mask.ptr<uint8_t>(y);
            cv::Vec3b* crow = color_mask.ptr<cv::Vec3b>(y);
            for (int x = 0; x < mask.cols; ++x) {
                const uint8_t cls = mrow[x];
                if (cls < 19) {
                    crow[x] = kCityscapesBgr[cls];
                }
            }
        }
    }

    return color_mask;
}

void MasksVisualizationEngine::displayOpenCVWindow(const cv::Mat& image, double latency_ms, double fps, double colorize_time_ms) {
    // Measure OpenCV display FPS
    auto current_time = std::chrono::steady_clock::now();
    opencv_frame_count_++;
    
    // Calculate FPS every 30 frames for smooth display
    if (opencv_frame_count_ % 30 == 0) {
        auto time_diff = std::chrono::duration<double>(current_time - last_opencv_time_).count();
        opencv_fps_ = 30.0 / time_diff;
        last_opencv_time_ = current_time;
    }
    
    // Add performance metrics overlay
    cv::Mat display_image = image.clone();
    
    // Performance metrics text
    std::string opencv_fps_text = "OpenCV FPS: " + std::to_string(static_cast<int>(opencv_fps_));
    std::string ros_fps_text = "ROS FPS: " + (fps > 0 ? std::to_string(static_cast<int>(fps)) : "N/A");
    std::string latency_text = "ROS Latency: " + (latency_ms > 0 ? std::to_string(static_cast<int>(latency_ms)) + " ms" : "N/A");
    std::string colorize_text = "Colorize: " + std::to_string(static_cast<int>(colorize_time_ms)) + " ms";
    std::string type_text = "Type: " + viz_type_;
    
    // Position text in top-left corner
    int y_offset = 30;
    int line_height = 40;
    
    cv::putText(display_image, opencv_fps_text, cv::Point(10, y_offset), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(display_image, ros_fps_text, cv::Point(10, y_offset + line_height), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(display_image, latency_text, cv::Point(10, y_offset + 2 * line_height), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(display_image, colorize_text, cv::Point(10, y_offset + 3 * line_height), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(display_image, type_text, cv::Point(10, y_offset + 4 * line_height), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
    
    // Display window
    cv::namedWindow("VisionPilot Visualization", cv::WINDOW_NORMAL);
    cv::resizeWindow("VisionPilot Visualization", 1280, 720);
    cv::imshow("VisionPilot Visualization", display_image);
    cv::waitKey(1);  // Non-blocking display update
}

} // namespace autoware_pov::common
