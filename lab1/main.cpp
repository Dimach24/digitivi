#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


cv::Mat getHist(const cv::Mat &image) {
    cv::Mat hist = cv::Mat::zeros(1, 256, CV_64FC1);


    for (int i = 0; i < image.cols; i++)
        for (int j = 0; j < image.rows; j++) {
            int r = image.at<unsigned char>(j, i);
            hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
        }

    double m = 0, M = 0;
    cv::minMaxLoc(hist, &m, &M);
    hist = hist / M;

    cv::Mat hist_img = cv::Mat::zeros(100, 256, CV_8U);

    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 100; j++) {
            if (hist.at<double>(0, i) * 100 > j) {
                hist_img.at<unsigned char>(99 - j, i) = 255;
            }
        }
    cv::bitwise_not(hist_img, hist_img);
    return hist_img;
}

cv::Mat downsample(const cv::Mat &in, size_t q) {
    size_t inWidth = in.cols, inHeight = in.rows;
    size_t outWidth = inWidth / q, outHeight = inHeight / q;

    cv::Mat out = cv::Mat::zeros(inHeight, inWidth, CV_8U);
    for (size_t i = 0; i < inHeight; ++i) {
        for (size_t j = 0; j < inWidth; ++j) {
            out.at<uchar>(i, j) = in.at<uchar>(i / q * q, j / q * q);
        }
    }
    return out;
}

cv::Mat quantise(const cv::Mat &in, size_t levelsNum) {
    const size_t inWidth = in.cols;
    const size_t inHeight = in.rows;
    const size_t qStep = UINT8_MAX / (levelsNum - 1);

    cv::Mat out = cv::Mat::zeros(inHeight, inWidth, CV_8U);

    for (size_t i = 0; i < inHeight; ++i) {
        for (size_t j = 0; j < inWidth; ++j) {
            const uint16_t shiftedL = in.at<uchar>(i, j) + qStep / 2;
            out.at<uchar>(i, j) = shiftedL / qStep * qStep;
        }
    }

    return out;
}

namespace Math {
double rms(const cv::Mat &in) {
    double sum = 0;
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            sum += pow(in.at<uchar>(i, j), 2);
        }
    }
    return sqrt(sum / in.cols / in.rows);
}

double mean(const cv::Mat &in) {
    double sum = 0;
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            sum += in.at<uchar>(i, j);
        }
    }
    return sum / in.cols / in.rows;
}

double std(const cv::Mat &in) {
    double mean_ = mean(in);
    double sum = 0;
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            sum += pow(in.at<uchar>(i, j) - mean_, 2);
        }
    }
    return sqrt(sum / in.cols / in.rows);
}

double std(const cv::Mat &a, const cv::Mat &b) {
    double sum = 0;
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            sum += pow(a.at<uchar>(i, j) - b.at<uchar>(i, j), 2);
        }
    }
    return sqrt(sum / a.cols / a.rows);
}
}

int main() {
    std::cout << "Starting application with OpenCV ver: " << CV_VERSION << "\n";

    const cv::Mat image = cv::imread(R"(resources/mordor.png)", 0);
    if (image.empty())
        return -1;

    const auto hist = getHist(image);
    auto downsampled = downsample(image, 4);
    auto quantised = quantise(image, 4);
    auto histQuantised = getHist(quantised);

    cv::imshow("Original", image);
    cv::imshow("Downsampled", downsampled);
    cv::imshow("HistogramOriginal", hist);
    cv::imshow("Quantised", quantised);
    cv::imshow("HistogramQuantised", histQuantised);

    std::cout << "Original RMS:\n\t" << Math::rms(image) << std::endl;
    for (size_t steps = 2; steps <= 64; steps <<= 1) {
        auto tmp = quantise(image, steps);
        std::cout << "========== " << steps << " ==========" << std::endl;
        std::cout << "Theoretical: " << static_cast<double>(UINT8_MAX / (steps - 1)) / sqrt(12.) <<
                std::endl << "Actual:      " << Math::std(image, tmp) << std::endl;
        auto tmpHist = getHist(tmp);
        cv::imwrite("quantised_in_" + std::to_string(steps) + "_steps.png", tmp);
        cv::imwrite("quantised_in_" + std::to_string(steps) + "_steps_hist.png", tmpHist);
    }

    for (auto &dsq: {4, 8}) {
        downsampled = downsample(image, dsq);
        cv::imwrite("downsampled_" + std::to_string(dsq) + "_times.png", downsampled);
    }
    cv::imwrite("orig.png", image);
    cv::imwrite("origHist.png", hist);
    cv::waitKey(0);
    return 0;
}
