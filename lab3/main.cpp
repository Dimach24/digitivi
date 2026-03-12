#include <cmath>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using Mask = std::vector<std::vector<float>>;

const Mask ROBERTS_X{{1, 0, 0}, {0, -1, 0}, {0, 0, 0}};
const Mask ROBERTS_Y{{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}};

Mask gaussFilter(int aperture, float r);
void filter(const cv::Mat &input_img, cv::Mat &output_img, const Mask &mask);

int main() {
    cv::Mat image = cv::imread(R"(D:\Stud\1107\DCH(Dymchenko_Chaminov)\LAB1\x64\Debug\ETU.jpg)", 0);
    if (image.empty())
        return -1;


    cv::resize(image, image, {640, 350});
    cv::Mat result = image.clone();
    Mask mask = gaussFilter(7, 2);
    filter(image, result, mask);
    cv::imshow("SRC", image);
    cv::imshow("GAUSS 3", result);
    filter(image, result, ROBERTS_X);
    cv::imshow("RobertsHorizontal", result);
    filter(image, result, ROBERTS_Y);
    cv::imshow("RobertsVertical", result);
    cv::waitKey(0);
}

Mask gaussFilter(int aperture, float r) {
    auto half = aperture / 2;
    std::vector<std::vector<float>> result(aperture, std::vector<float>(aperture));
    for (int i = -half; i <= half; ++i)
        for (int j = -half; j <= half; ++j) {
            result[i + half][j + half] = std::exp(-(i * i + j * j) / 2. / r / r);
        }
    return result;
}

void filter(const cv::Mat &input_img, cv::Mat &output_img, const Mask &mask) {
    output_img = cv::Mat::zeros(input_img.size(), CV_8U);
    float k = 0.0f;
    for (auto &row: mask) {
        for (auto &element: row) {
            k += element;
        }
    }
    if (abs(k) < 1e-6) {
        k = 1;
    }

    size_t aperture = mask.size();
    int half = aperture / 2;

    for (int y = half; y < input_img.rows - half; ++y) {
        for (int x = half; x < input_img.cols - half; ++x) {
            float rez = 0.0f;

            for (int dy = -half; dy <= half; ++dy) {
                for (int dx = -half; dx <= half; ++dx) {
                    uchar pixel = input_img.at<uchar>(y + dy, x + dx);
                    rez += mask[dy + half][dx + half] * pixel;
                }
            }

            output_img.at<uchar>(y, x) = static_cast<uchar>(rez / k);
        }
    }
}
