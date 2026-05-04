#include <filesystem>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

struct StructElem {
    std::vector<std::vector<int>> mask;
    int ax;
    int ay;
};

static const StructElem square3{
        std::vector{std::vector{1, 1, 1}, std::vector{1, 1, 1}, std::vector{1, 1, 1}},
        1,
        1,
};
static const StructElem square5{
        std::vector(5, std::vector(5, 1)),
        2,
        2,
};
static const StructElem square7{
        std::vector(7, std::vector(7, 1)),
        3,
        3,
};
static const StructElem cross3{
        std::vector{std::vector{0, 1, 0}, std::vector{1, 1, 1}, std::vector{0, 1, 0}},
        1,
        1,
};
static const StructElem line{
        std::vector{std::vector(3, 1), std::vector(3, 0), std::vector(3, 0)},
        0,
        1,
};
void erosion(const cv::Mat &input, cv::Mat &output, const StructElem &se);
void dilatation(const cv::Mat &input, cv::Mat &output, const StructElem &se);
void opening(const cv::Mat &input, cv::Mat &output, const StructElem &se);
void closing(const cv::Mat &input, cv::Mat &output, const StructElem &se);
void msmg(const cv::Mat &input, cv::Mat &output, const std::vector<StructElem> &masks);
void showAndSave(const std::string &name, const cv::Mat &image) {
    static const std::filesystem::path outPath{std::filesystem::current_path() / "out"};
    if (!std::filesystem::exists(outPath)) {
        std::filesystem::create_directory(outPath);
    }
    cv::imshow(name, image);
    cv::imwrite((outPath / (name + ".png")).string(), image);
}

int main() {
    const cv::Mat source = cv::imread(R"(resources/1280pxEtu.jpg)", 0);
    cv::Mat image;
    constexpr auto scale = 2;
    cv::resize(source, image, {source.cols / scale, source.rows / scale});
    if (image.empty())
        return EXIT_FAILURE;

    showAndSave("src", image);
    uchar threshold = static_cast<uchar>(cv::mean(image)[0] * .707);
    auto bin = image.clone();
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            bin.at<uchar>(i, j) = 255 * (image.at<uchar>(i, j) > threshold);
        }
    }
    showAndSave("binarized", bin);
    cv::Mat processed = bin.clone();

    for (const auto &[name, mask]: std::vector<std::pair<std::string, StructElem>>{
                 {"3x3 square", square3},
                 {"7x7 square", square7},
                 {"3x3 cross", cross3},
                 {"1x3 line", line},
         }) {
        erosion(bin, processed, mask);
        showAndSave("Eroded (bin) " + name, processed);
        dilatation(bin, processed, mask);
        showAndSave("Dilated (bin) " + name, processed);
        opening(bin, processed, mask);
        showAndSave("Opened (bin) " + name, processed);
        closing(bin, processed, mask);
        showAndSave("Closed (bin) " + name, processed);
    }


    for (const auto &[name, mask]:
         std::vector<std::pair<std::string, StructElem>>{
                 {"3x3 square", square3},
                 {"7x7 square", square7},
         })

    {
        erosion(image, processed, mask);
        showAndSave("Eroded " + name, processed);
        dilatation(image, processed, mask);
        showAndSave("Dilated " + name, processed);
        opening(image, processed, mask);
        showAndSave("Opened " + name, processed);
        closing(image, processed, mask);
        showAndSave("Closed " + name, processed);
    }

    {
        cv::Mat eroded2 = cv::Mat::zeros(image.size(), CV_8U);
        cv::Mat dilated2 = cv::Mat::zeros(image.size(), CV_8U);
        dilatation(image, processed, square3);
        dilatation(processed, dilated2, square3);
        erosion(image, processed, square3);
        erosion(image, eroded2, square3);
        showAndSave("Contours L-2eroded", image - eroded2);
        showAndSave("Contours 2dilated-L", dilated2 - image);
        showAndSave("Contours 2dilated-2eroded", dilated2 - eroded2);
        msmg(image, processed, {square3, square5, square7});
        showAndSave("Multi-Scale Morphological Gradient", processed);
    }
    {
        cv::Mat eroded2 = cv::Mat::zeros(image.size(), CV_8U);
        cv::Mat dilated2 = cv::Mat::zeros(image.size(), CV_8U);
        dilatation(image, processed, cross3);
        dilatation(processed, dilated2, cross3);
        erosion(image, processed, cross3);
        erosion(image, eroded2, cross3);
        showAndSave("Contours L-2eroded cross", image - eroded2);
        showAndSave("Contours 2dilated-L cross", dilated2 - image);
        showAndSave("Contours 2dilated-2eroded cross", dilated2 - eroded2);
    }

    cv::waitKey();


    return EXIT_SUCCESS;
}

void erosion(const cv::Mat &input, cv::Mat &output, const StructElem &se) {
    if (input.empty() || se.mask.empty() || se.mask[0].empty()) {
        return;
    }
    const int mh = static_cast<int>(se.mask.size());
    const int mw = (mh > 0) ? static_cast<int>(se.mask[0].size()) : 0;

    // Чтобы не итерироваться по нулям
    std::vector<std::pair<int, int>> maskPoints;
    for (int j = 0; j < mh; j++) {
        for (int i = 0; i < mw; i++) {
            if (se.mask[j][i]) {
                maskPoints.push_back(std::make_pair(i - se.ax, j - se.ay));
            }
        }
    }
    output = cv::Mat::zeros(input.size(), CV_8U);

    for (int y = se.ay; y < input.rows - (mh - se.ay - 1); y++) {
        for (int x = se.ax; x < input.cols - (mw - se.ax - 1); x++) {
            uchar pixelVal = UCHAR_MAX;
            for (const auto &[dx, dy]: maskPoints) {
                pixelVal = std::min(input.at<uchar>(y + dy, x + dx), pixelVal);
            }
            output.at<uchar>(y, x) = pixelVal;
        }
    }
}

void dilatation(const cv::Mat &input, cv::Mat &output, const StructElem &se) {
    if (input.empty() || se.mask.empty() || se.mask[0].empty()) {
        return;
    }
    const int mh = static_cast<int>(se.mask.size());
    const int mw = (mh > 0) ? static_cast<int>(se.mask[0].size()) : 0;

    // Чтобы не итерироваться по нулям
    std::vector<std::pair<int, int>> maskPoints;
    for (int j = 0; j < mh; j++) {
        for (int i = 0; i < mw; i++) {
            if (se.mask[j][i]) {
                maskPoints.push_back(std::make_pair(i - se.ax, j - se.ay));
            }
        }
    }
    output = cv::Mat::zeros(input.size(), CV_8U);

    for (int y = se.ay; y < input.rows - (mh - se.ay - 1); y++) {
        for (int x = se.ax; x < input.cols - (mw - se.ax - 1); x++) {
            uchar pixelVal = 0;
            for (const auto &[dx, dy]: maskPoints) {
                pixelVal = std::max(input.at<uchar>(y + dy, x + dx), pixelVal);
            }
            output.at<uchar>(y, x) = pixelVal;
        }
    }
}

void opening(const cv::Mat &input, cv::Mat &output, const StructElem &se) {
    cv::Mat tmp = cv::Mat::zeros(input.size(), CV_8U);
    output = cv::Mat::zeros(input.size(), CV_8U);
    erosion(input, tmp, se);
    dilatation(tmp, output, se);
}
void closing(const cv::Mat &input, cv::Mat &output, const StructElem &se) {
    cv::Mat tmp = cv::Mat::zeros(input.size(), CV_8U);
    output = cv::Mat::zeros(input.size(), CV_8U);
    dilatation(input, tmp, se);
    erosion(tmp, output, se);
}

void msmg(const cv::Mat &input, cv::Mat &output, const std::vector<StructElem> &masks) {
    cv::Mat res = cv::Mat::zeros(input.size(), CV_32F);
    for (int i = 0; i < masks.size(); ++i) {
        cv::Mat dilated = cv::Mat::zeros(input.size(), CV_8U);
        cv::Mat eroded = cv::Mat::zeros(input.size(), CV_8U);
        cv::Mat erodedPrev = cv::Mat::zeros(input.size(), CV_8U);
        dilatation(input, dilated, masks[i]);
        erosion(input, eroded, masks[i]);
        if (i > 0) {
            erosion(dilated - eroded, erodedPrev, masks[i - 1]);
        } else {
            erodedPrev = dilated - eroded;
        }
        res += erodedPrev / static_cast<float>(masks.size());
    }
    res.convertTo(output, CV_8U);
}
