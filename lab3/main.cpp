#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using Mask = std::vector<std::vector<float>>;

const Mask ROBERTS_X{
        {1, 0},
        {0, -1},
};
const Mask ROBERTS_Y{
        {0, 1},
        {-1, 0},
};
const Mask APERTURE_CORRECTION{
        {-1, -1, -1},
        {-1, 20, -1},
        {-1, -1, -1},
};

cv::Mat mosaicFilter(const cv::Mat &in, int blockSize);
cv::Mat medianFilter(const cv::Mat &in, int blockSize);
Mask gaussFilter(int aperture, float r);
void filter(const cv::Mat &input_img, cv::Mat &output_img, const Mask &mask);

int main() {
    cv::Mat image = cv::imread(R"(resources/Chess.jpg)", 0);
    if (image.empty())
        return -1;

    constexpr size_t optimalWidth = 500;
    const size_t scale = image.cols / optimalWidth;
    cv::resize(image, image, cv::Size(image.cols / scale, image.rows / scale));

    cv::Mat result = image.clone();
    Mask mask = gaussFilter(3, 1);
    filter(image, result, mask);
    cv::imshow("Source", image);
    cv::imwrite("source.png", result);

    cv::imshow("Gauss 3", result);
    cv::imwrite("gauss-3x3-1.png", result);
    mask = gaussFilter(7, 2);
    filter(image, result, mask);
    cv::imshow("Gauss 7", result);
    cv::imwrite("gauss-7x7-2.png", result);

    result = mosaicFilter(image, 5);
    cv::imshow("Mosaic", result);
    cv::imwrite("mosaic-5x5.png", result);
    filter(image, result, APERTURE_CORRECTION);
    cv::imshow("Aperture Correction", result);
    cv::imwrite("ac-20.png", result);

    result = medianFilter(image, 3);
    cv::imshow("Median", result);
    cv::imwrite("median-3x3.png", result);

    cv::Canny(image, result, 200, 250);
    cv::imshow("Canny", result);
    cv::imwrite("canny.png", result);


    filter(image, result, ROBERTS_X);
    cv::imshow("Roberts Horizontal", result);
    cv::imwrite("rh.png", result);

    filter(image, result, ROBERTS_Y);
    cv::imshow("Roberts Vertical", result);
    cv::imwrite("rv.png", result);

    cv::waitKey(0);
    return 0;
}


cv::Mat mosaicFilter(const cv::Mat &in, int blockSize) {
    cv::Mat result = in.clone();
    for (int y = 0; y + blockSize <= in.rows; y += blockSize) {
        for (int x = 0; x + blockSize <= in.cols; x += blockSize) {
            cv::Rect rect = cv::Rect(x, y, blockSize, blockSize);
            cv::Mat tmp;
            in(rect).convertTo(tmp, CV_32F);
            float rez = cv::sum(tmp)[0] / blockSize / blockSize;
            result(rect) = rez;
        }
    }
    return result;
}
cv::Mat medianFilter(const cv::Mat &in, int blockSize) {
    cv::Mat result = in.clone();
    std::vector<uchar> blockData(blockSize * blockSize);
    for (int y = 0; y + blockSize <= in.rows; ++y) {
        for (int x = 0; x + blockSize <= in.cols; ++x) {
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    blockData[i * blockSize + j] = in.at<uchar>(y + i, x + j);
                }
            }
            std::sort(blockData.begin(), blockData.end());
            float left = blockData[(blockData.size() + 1) / 2];
            float right = blockData[blockData.size() / 2 + 1];
            cv::Rect rect = cv::Rect(x, y, blockSize, blockSize);
            result(rect) = static_cast<uchar>((left + right) / 2);
        }
    }
    return result;
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
    output_img = input_img.clone();
    float k = 0.0f;
    for (const auto &row: mask) {
        for (const auto &element: row) {
            k += element;
        }
    }
    if (std::abs(k) < 1e-6) {
        k = 1.0f;
    }

    const int aperture = static_cast<int>(mask.size());
    const int half = aperture / 2;
    const int start = half;
    const int stop = half - 1 + aperture % 2; // Для нечетных симметрично

    for (int y = start; y < input_img.rows - stop; ++y) {
        for (int x = start; x < input_img.cols - stop; ++x) {
            float rez = 0.0f;
            for (int dy = -start; dy <= stop; ++dy) {
                for (int dx = -start; dx <= stop; ++dx) {
                    uchar pixel = input_img.at<uchar>(y + dy, x + dx);
                    rez += mask[dy + half][dx + half] * static_cast<float>(pixel);
                }
            }

            output_img.at<uchar>(y, x) = static_cast<uchar>(rez / k);
        }
    }
}
