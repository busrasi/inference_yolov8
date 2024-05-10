#pragma once
#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{ 0 };
    std::string className{};
    float confidence{ 0.0 };
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape = { 1280, 1280 }, const std::string& classesTxtFile = "", const bool& runWithCuda = true);
    std::vector<Detection> runInference(const cv::Mat& input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat& source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};

    std::vector<std::string> classes = {
    "2c_s", "2d_s", "2h_s", "2s_s",
    "3c_s", "3d_s", "3h_s", "3s_s",
    "4c_s", "4d_s", "4h_s", "4s_s",
    "5c_s", "5d_s", "5h_s", "5s_s",
    "6c_s", "6d_s", "6h_s", "6s_s",
    "7c_s", "7d_s", "7h_s", "7s_s",
    "8c_s", "8d_s", "8h_s", "8s_s",
    "9c_s", "9d_s", "9h_s", "9s_s",
    "Tc_s", "Td_s", "Th_s", "Ts_s",
    "Jc_s", "Jd_s", "Jh_s", "Js_s",
    "Qc_s", "Qd_s", "Qh_s", "Qs_s",
    "Kc_s", "Kd_s", "Kh_s", "Ks_s",
    "Ac_s", "Ad_s", "Ah_s", "As_s",
    "chips"
    };

    cv::Size2f modelShape{};

    float modelConfidenseThreshold{ 0.25 };
    float modelScoreThreshold{ 0.45 };
    float modelNMSThreshold{ 0.50 };

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

#endif // INFERENCE_H