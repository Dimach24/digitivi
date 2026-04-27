#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

cv::Mat buildLut(const std::function<float(float)> &T);
cv::Mat getHist(const cv::Mat &image);
cv::Mat ssr(const cv::Mat &img, float sigma, size_t aperture);
cv::Mat
msr(const cv::Mat &img, float sigma0, const std::vector<float> &scales, std::vector<float> weights);
void saveDataForReport(const cv::Mat &image, const std::string &tag);

int main() {
    cv::Mat image = cv::imread(R"(resources/evening.jpg)", 0);
    if (image.empty())
        return -1;
    cv::Mat hist;
    cv::Mat lut;

    cv::imshow("src", image);
    hist = getHist(image);
    cv::imshow("src hist", hist);
    saveDataForReport(image, "src");

    {
        double minVal, maxVal;
        cv::minMaxLoc(image, &minVal, &maxVal);
        lut = buildLut([minVal, maxVal](const float &x) {
            return (x * 255 - minVal) / (maxVal - minVal);
        });
        cv::Mat stretched;
        cv::LUT(image, lut, stretched);
        cv::imshow("stretch", stretched);
        hist = getHist(stretched);
        cv::imshow("stretch hist", hist);
        saveDataForReport(stretched, "stretch");
    }

    {
        lut = buildLut([](const float &x) { return 1 - x; });
        cv::Mat inversed;
        cv::LUT(image, lut, inversed);
        cv::imshow("negative", inversed);
        hist = getHist(inversed);
        cv::imshow("neg hist", hist);
        saveDataForReport(inversed, "neg");
    }

    {
        auto constexpr c = 1.f;
        auto constexpr gamma = .45f;
        lut = buildLut([](const float &x) { return c * cv::pow(x, gamma); });
        cv::Mat gammaCorr;
        cv::LUT(image, lut, gammaCorr);

        cv::imshow("gamma", gammaCorr);
        hist = getHist(gammaCorr);
        cv::imshow("gamma hist", hist);
        saveDataForReport(gammaCorr, "gamma");
    }

    {
        float const c = 1.f / cv::log(2.f);
        lut = buildLut([c](const float &x) { return c * cv::log(1 + x); });
        cv::Mat logCorr;
        cv::LUT(image, lut, logCorr);

        cv::imshow("log", logCorr);
        hist = getHist(logCorr);
        cv::imshow("log hist", hist);
        saveDataForReport(logCorr, "log");
    }


    {
        cv::Mat src = image.clone();
        cv::Mat msrRes = msr(src, 5, {1, 10, 20}, std::vector<float>(3, 1. / 3));
        cv::imshow("msr", msrRes);
        hist = getHist(msrRes);
        cv::imshow("msr hist", hist);
        saveDataForReport(msrRes, "msr");
    }

    cv::waitKey();
    return EXIT_SUCCESS;
}


cv::Mat buildLut(const std::function<float(float)> &T) {
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        float f = static_cast<float>(i) / 255.0f;
        float g = T(f);
        int v = static_cast<int>(std::lround(g * 255.0f));
        lut.at<uchar>(0, i) = cv::saturate_cast<uchar>(v);
    }
    return lut;
}
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
cv::Mat ssr(const cv::Mat &img, float sigma, size_t aperture) {
    cv::Mat conv = img.clone();
    cv::GaussianBlur(
            img,
            conv,
            cv::Size(static_cast<int>(aperture), static_cast<int>(aperture)),
            sigma,
            sigma);
    for (int i = 0; i < conv.rows; ++i) {
        for (int j = 0; j < conv.cols; ++j) {
            if (conv.at<float>(i, j) < 1e-6) {
                conv.at<float>(i, j) = 1e-6;
            }
        }
    }
    cv::Mat logConv;
    cv::log(conv, logConv);
    cv::Mat src;
    cv::log(img, src);
    return src - logConv;
}
cv::Mat
msr(const cv::Mat &img,
    float sigma0,
    const std::vector<float> &scales,
    std::vector<float> weights) {
    cv::Mat src;
    img.convertTo(src, CV_32F);
    src /= 255;
    if (weights.empty()) {
        weights.resize(scales.size());
        for (auto &el: weights) {
            el = 1.f / static_cast<float>(scales.size());
        }
    }

    cv::Mat sum = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    for (size_t i = 0; i < scales.size(); ++i) {
        auto sigma = sigma0 * scales[i];
        auto aperture = static_cast<size_t>(3.f * sigma);
        aperture = aperture / 2 * 2 + 1;
        sum += weights[i] * ssr(src, sigma, aperture);
    }
    cv::Mat result = img.clone();
    double minVal, maxVal;

    // Сортируем все значения sum для поиска процентилей
    cv::Mat flat = sum.reshape(1, 1); // превращаем в 1D
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

    int total = sorted.cols;
    int p1_idx = static_cast<int>(0.01 * total); // 1-й процентиль
    int p99_idx = static_cast<int>(0.99 * total); // 99-й процентиль

    double low = sorted.at<float>(p1_idx);
    double high = sorted.at<float>(p99_idx);

    if (high <= low) {
        high = low + 1e-6;
    }

    // Normalize с отсечением выбросов
    cv::Mat normalized;
    sum = (sum - low) / (high - low) * 255;
    cv::Mat result8u;
    sum.convertTo(result8u, CV_8U);

    return result8u;
};
void saveDataForReport(const cv::Mat &image, const std::string &tag) {
    cv::Scalar mean, stdDev;
    cv::meanStdDev(image, mean, stdDev);
    std::cout << tag << " \tstd: " << stdDev[0] << std::endl;
    cv::imwrite(tag + ".png", image);
    auto hist = getHist(image);
    cv::imwrite(tag + ".hist.png", hist);
}
