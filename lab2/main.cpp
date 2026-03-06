#include <cmath>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


struct UserData {
    const cv::Mat &picture;
    const cv::Mat &pictureWithGrid;
    int blockSize;
    const char *winName;
};

cv::Mat addGrid(const cv::Mat &input, int blockSize);
void onMouse(int event, int x, int y, int flags, void *userData);
cv::Mat dctBasis(int blockSize);
cv::Mat transform(const cv::Mat &input);
void showScaled(const cv::Mat &toShow, std::string winName, int totalHeight = 256);
int main() {
    std::cout << "Starting application with OpenCV ver: " << CV_VERSION << "\n";

    const cv::Mat image = cv::imread(R"(resources/mordor.png)", 0);
    if (image.empty())
        return -1;
    const char winName[]{"Main"};
    cv::namedWindow(winName);

    auto blockSize = 8;
    auto result = addGrid(image, blockSize);

    UserData data{image, result, blockSize, winName};

    cv::setMouseCallback(winName, onMouse, &data);

    imshow(winName, result);

    cv::waitKey(0);

    return 0;
}

cv::Mat addGrid(const cv::Mat &input, int blockSize) {
    cv::Mat result;
    cv::cvtColor(input, result, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < result.rows / blockSize; ++i)
        for (int j = 0; j < result.cols / blockSize; ++j) {
            cv::rectangle(
                    result,
                    cv::Rect{j * blockSize, i * blockSize, blockSize, blockSize},
                    cv::Scalar(255, 255, 0),
                    1);
        }
    return result;
}

void onMouse(int event, int x, int y, int flags, void *userData) {
    auto data = *static_cast<UserData *>(userData);

    if (event == cv::EVENT_LBUTTONDOWN) {
        int i = y / data.blockSize * data.blockSize;
        int j = x / data.blockSize * data.blockSize;
        if (i + data.blockSize >= data.picture.rows || j + data.blockSize >= data.picture.cols) {
            return;
        }
        auto gridCopy = data.pictureWithGrid.clone();
        auto blockRect = cv::Rect{j, i, data.blockSize, data.blockSize};
        cv::rectangle(gridCopy, blockRect, cv::Scalar{255, 0, 255}, 2);
        imshow(data.winName, gridCopy);
        auto block = data.picture(blockRect);
        showScaled(block, "Block");
        auto DFT = transform(block);
        showScaled(DFT, "DFT");
    }
}

cv::Mat dctBasis(int blockSize) {
    cv::Mat basisMat = cv::Mat::zeros(blockSize, blockSize, CV_64F) + 1 / std::sqrt(blockSize);
    const double sqrt2 = std::sqrt(2.);
    for (int n = 1; n < blockSize; ++n) {
        for (int k = 0; k < blockSize; ++k) {
            basisMat.at<double>(n, k) *= sqrt2 * std::cos(CV_PI * n / blockSize * (k + 0.5));
        }
    }
    return basisMat;
}
cv::Mat transform(const cv::Mat &input) {
    assert(input.channels() == 1);
    assert(input.rows == input.cols);
    cv::Mat U;
    if (input.type() != CV_64F) {
        input.convertTo(U, CV_64F);
    } else {
        U = input.clone();
    }
    auto N = input.rows;
    auto phi = dctBasis(N);
    return phi * U * phi.t();
}
void showScaled(const cv::Mat &toShow, std::string winName, int totalHeight) {
    auto rows = toShow.rows;
    auto scale = totalHeight / rows;
    cv::Mat vis;
    toShow.convertTo(vis, CV_8U);
    cv::resize(vis, vis, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(winName, vis);
}
