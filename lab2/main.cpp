#include <cmath>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

struct CallbackMouseEventData {
    const cv::Mat &image;
    int blockSize;
    const std::vector<cv::Rect> &blocks;
};

std::vector<cv::Rect> splitToBlocks(size_t blockSize, size_t height, size_t width);

cv::Mat drawGrid(const cv::Mat &image, const std::vector<cv::Rect> &blocks) {
    auto vis = image.clone();
    for (const auto &block: blocks) {
        cv::rectangle(vis, block, cv::Scalar{0, 255, 0}, 1);
    }
    return vis;
}

cv::Mat getBasisMat(int N) {
    static cv::Mat basisMat;
    if (basisMat.rows == N) {
        return basisMat;
    }
    basisMat = cv::Mat::zeros(N, N, CV_64F);
    basisMat += 1 / std::sqrt(N);
    double sqrt2 = std::sqrt(2);
    for (int n = 1; n < N; n++) {
        for (int k = 0; k < N; k++) {
            basisMat.at<double>(n, k) *= sqrt2 * std::cos(CV_PI * n / N * (k + .5));
        }
    }
    return basisMat;
}
cv::Mat getTransformMat(cv::Mat &U, int N) {
    cv::Mat basisMat = getBasisMat(N);
    return basisMat * U * basisMat.t();
}

void mouseClickHandler(int x, int y, int flags, void *userdata) {
    auto &data = *static_cast<CallbackMouseEventData *>(userdata);
    const int blockSize = data.blockSize;
    int blockX = x / blockSize * blockSize;
    int blockY = y / blockSize * blockSize;
    if (blockX + blockSize > data.image.cols || blockY + blockSize > data.image.rows) {
        return;
    }
    auto vis = drawGrid(data.image, data.blocks);

    cv::Rect rect{blockX, blockY, blockSize, blockSize};
    cv::rectangle(vis, rect, cv::Scalar{0, 0, 255}, 1);
    cv::imshow("with_grid", vis);

    cv::Mat block;
    cv::cvtColor(data.image(rect), block, cv::COLOR_BGR2GRAY);
    cv::Mat U;
    block.convertTo(U, CV_64F);
    auto dct = getTransformMat(U, blockSize);
    cv::Mat dct8u;
    dct.convertTo(dct8u, CV_8U);
    cv::Mat dctVis;
    auto scale = 8 * 32 / blockSize;
    cv::resize(dct8u, dctVis, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow("DCT", dctVis);
    cv::Mat blockVis;
    cv::resize(block, blockVis, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow("block", blockVis);
}


int main() {
    std::cout << "Starting application with OpenCV ver: " << CV_VERSION << "\n";

    constexpr size_t blockSize = 8;

    const cv::Mat image = cv::imread(R"(resources/1280pxEtu.jpg)");
    if (image.empty())
        return EXIT_FAILURE;


    auto blocks = splitToBlocks(blockSize, image.rows, image.cols);

    cv::namedWindow("with_grid", cv::WINDOW_FULLSCREEN);
    CallbackMouseEventData callbackData{image, blockSize, blocks};
    cv::setMouseCallback(
            "with_grid",
            [](int event, int x, int y, int flags, void *userdata) -> void {
                if (event == cv::EVENT_LBUTTONDOWN)
                    mouseClickHandler(x, y, flags, userdata);
            },
            &callbackData);
    cv::Mat vis = drawGrid(image, blocks);
    cv::imshow("with_grid", vis);

    cv::imwrite("with_grid.png", vis);
    cv::waitKey();
    return 0;
}
std::vector<cv::Rect> splitToBlocks(size_t blockSize, size_t height, size_t width) {
    std::vector<cv::Rect> blocks;
    blocks.reserve(width / blockSize * (height / blockSize));
    for (size_t i = 0; i < height / blockSize; i++) {
        for (size_t j = 0; j < width / blockSize; j++) {
            blocks.emplace_back(j * blockSize, i * blockSize, blockSize, blockSize);
        }
    }
    return blocks;
}
