#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


double entropyByProbs(const std::vector<double> &probabilities) {
    double result = 0.;
    for (auto &prob: probabilities) {
        if (prob > 0) { // для не встретившихся значений
            result -= prob * std::log2(prob);
        }
    }
    return result;
}
double imageEntropy(const std::vector<uchar> &pixels) {
    std::vector<size_t> histogram(UCHAR_MAX + 1, 0);
    std::vector<double> probabilities(UCHAR_MAX + 1, 0);
    for (auto &pxVal: pixels) {
        histogram[static_cast<size_t>(pxVal)]++;
    }
    for (int i = 0; i < histogram.size(); i++) {
        probabilities[i] = static_cast<double>(histogram[i]) / static_cast<double>(pixels.size());
    }
    return entropyByProbs(probabilities);
}

cv::Mat shift(const cv::Mat &image) {
    assert(image.type() == CV_8U);

    cv::Mat shifted(image.rows, image.cols, CV_8S);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            shifted.at<char>(i, j) = static_cast<char>(image.at<uchar>(i, j) - 128l);
        }
    }
    return shifted;
}
cv::Mat unshift(const cv::Mat &image) {
    assert(image.type() == CV_8S);
    cv::Mat unshifted(image.rows, image.cols, CV_8U);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            unshifted.at<uchar>(i, j) = static_cast<uchar>(image.at<char>(i, j) + 128l);
        }
    }
    return unshifted;
}

// Из 2 работы
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
cv::Mat transformImage(const cv::Mat &image, int blockSize) {
    assert(image.type() == CV_8S);
    assert(image.rows % blockSize == 0);
    assert(image.cols % blockSize == 0);

    cv::Mat transformed(image.rows, image.cols, CV_64F, cv::Scalar(0));
    for (int i = 0; (i + 1) * blockSize <= image.rows; i++) {
        for (int j = 0; (j + 1) * blockSize <= image.cols; j++) {
            cv::Rect rect(j * blockSize, i * blockSize, blockSize, blockSize);
            transform(image(rect)).copyTo(transformed(rect));
        }
    }
    return transformed;
}

cv::Mat reverseTransform(const cv::Mat &input) {
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
    return phi.t() * U * phi;
}
cv::Mat reverseTransformImage(const cv::Mat &image, int blockSize) {
    assert(image.type() == CV_64F);
    assert(image.rows % blockSize == 0);
    assert(image.cols % blockSize == 0);
    cv::Mat transformed(image.rows, image.cols, CV_8S, cv::Scalar(0));
    for (int i = 0; (i + 1) * blockSize <= image.rows; i++) {
        for (int j = 0; (j + 1) * blockSize <= image.cols; j++) {
            cv::Rect rect(j * blockSize, i * blockSize, blockSize, blockSize);
            reverseTransform(image(rect)).convertTo(transformed(rect), CV_8S);
        }
    }
    return transformed;
}

cv::Mat tableByQuality(uchar quality) {
    assert(quality <= 25);
    assert(quality >= 1);
    cv::Mat result(8, 8, CV_8U);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            result.at<uint8_t>(i, j) = 8 + (i + j) * quality;
        }
    }
    return result;
}
cv::Mat quantiseBlock(const cv::Mat &input, const cv::Mat &table) {
    assert(input.rows == input.cols);
    assert(input.rows == table.rows);
    assert(input.cols == table.cols);
    assert(input.type() == CV_64F);
    assert(table.type() == CV_8U);
    cv::Mat result(input.rows, input.cols, CV_8S);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result.at<char>(i, j) =
                    cv::saturate_cast<char>(input.at<double>(i, j) / table.at<uchar>(i, j));
        }
    }
    return result;
}
cv::Mat quantise(const cv::Mat &image, int blockSize, int quality) {
    assert(image.channels() == 1);
    assert(image.rows % blockSize == 0);
    assert(image.cols % blockSize == 0);

    cv::Mat table = tableByQuality(quality);
    cv::Mat quantised(image.size(), CV_8S, cv::Scalar(0));
    for (int i = 0; (i + 1) * blockSize <= image.rows; i++) {
        for (int j = 0; (j + 1) * blockSize <= image.cols; j++) {
            cv::Rect rect(j * blockSize, i * blockSize, blockSize, blockSize);
            quantiseBlock(image(rect), table).copyTo(quantised(rect));
        }
    }
    return quantised;
}

cv::Mat dequantiseBlock(const cv::Mat &input, const cv::Mat &table) {
    assert(input.rows == input.cols);
    assert(input.rows == table.rows);
    assert(input.cols == table.cols);
    assert(input.type() == CV_8S);
    assert(table.type() == CV_8U);
    cv::Mat result(input.rows, input.cols, CV_64F);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result.at<double>(i, j) =
                    static_cast<double>(input.at<char>(i, j)) * table.at<uchar>(i, j);
        }
    }
    return result;
}
cv::Mat dequantise(const cv::Mat &quantised, int blockSize, int quality) {
    assert(quantised.channels() == 1);
    assert(quantised.rows % blockSize == 0);
    assert(quantised.cols % blockSize == 0);

    cv::Mat table = tableByQuality(quality);
    cv::Mat result(quantised.size(), CV_64F, cv::Scalar(0));
    for (int i = 0; (i + 1) * blockSize <= quantised.rows; i++) {
        for (int j = 0; (j + 1) * blockSize <= quantised.cols; j++) {
            cv::Rect rect(j * blockSize, i * blockSize, blockSize, blockSize);
            dequantiseBlock(quantised(rect), table).copyTo(result(rect));
        }
    }
    return result;
}

std::vector<std::pair<int, int>> zigzagOrder(int blockSize) {
    std::vector<std::pair<int, int>> order;
    order.reserve(blockSize * blockSize);
    for (int s = 0; s < blockSize; s++) {
        if (s % 2 == 0) { // четная диагональ: сверху вниз
            for (int i = s; i >= 0; i--) {
                order.emplace_back(i, s - i);
            }
        } else { // нечетная диагональ: снизу вверх
            for (int i = 0; i <= s; i++) {
                order.emplace_back(i, s - i);
            }
        }
    }

    for (int s = blockSize; s < 2 * blockSize - 1; s++) {
        if (s % 2 == 0) { // четная диагональ
            for (int i = blockSize - 1; i >= s - blockSize + 1; i--) {
                order.emplace_back(i, s - i);
            }
        } else { // нечетная диагональ
            for (int i = s - blockSize + 1; i < blockSize; i++) {
                order.emplace_back(i, s - i);
            }
        }
    }

    return order;
}
std::vector<char> zigZagRead(const cv::Mat &coefs, int blockSize) {
    assert(coefs.type() == CV_8S);
    std::vector<char> zigZag(coefs.rows * coefs.cols, 0);
    size_t k = 0;
    auto order = zigzagOrder(blockSize);
    for (int i = 0; (i + 1) * blockSize <= coefs.rows; i++) {
        for (int j = 0; (j + 1) * blockSize <= coefs.cols; j++) {
            for (const auto &[di, dj]: order) {
                zigZag[k++] = coefs.at<char>(i * blockSize + di, j * blockSize + dj);
            }
        }
    }
    return zigZag;
}
cv::Mat zigZagWrite(const std::vector<char> &zigZag, int blockSize, int iw, int ih) {
    assert(zigZag.size() % (blockSize * blockSize) == 0);
    assert(zigZag.size() == iw * ih);
    cv::Mat result(ih, iw, CV_8S);
    size_t k = 0;
    auto order = zigzagOrder(blockSize);
    for (int i = 0; (i + 1) * blockSize <= ih; i++) {
        for (int j = 0; (j + 1) * blockSize <= iw; j++) {
            for (const auto &[di, dj]: order) {
                result.at<char>(i * blockSize + di, j * blockSize + dj) = zigZag[k++];
            }
        }
    }
    return result;
}

const std::filesystem::path outPath(std::filesystem::current_path() / "out");


union RleElement {
    struct {
        uchar run = 0;
        char value = 0;
    } ac;
    char dc;

    explicit RleElement(char value) : dc(value) {
    }
    explicit RleElement(uchar run, char value) : ac{run, value} {
    }
    bool EOB() const {
        return ac.run == 0 && ac.value == 0;
    }
};

std::vector<RleElement> runLengthEncoding(const std::vector<char> &sequence, const int blockSize) {
    std::vector<RleElement> rleEncoded;
    char lastDc = 0;
    uchar counter = 0;
    auto blockSize2 = blockSize * blockSize;
    for (size_t i = 0; i < sequence.size(); i++) {
        uchar blockIdx = i % blockSize2;
        if (blockIdx == 0) {
            rleEncoded.emplace_back(sequence[i] - lastDc);
            lastDc = sequence[i];
            counter = 0;
            continue;
        }
        if (sequence[i]) {
            rleEncoded.emplace_back(counter, sequence[i]);
            counter = 0;
            continue;
        }
        if (blockIdx == blockSize2 - 1) {
            rleEncoded.emplace_back(0, 0); // EOB, после Хаффмана будет 0b10
            counter = 0;
            continue;
        }
        counter++;
    }
    return rleEncoded;
}
std::pair<std::vector<char>, size_t>
runLengthDecoding(const std::vector<RleElement> &encoded, const int blockSize) {
    std::vector<char> decoded;
    size_t bitLength = 0;
    int lastDc = 0;
    const uchar blockSize2 = blockSize * blockSize;
    auto numBlocks = 0;
    bool isDc = true;
    for (const auto &element: encoded) {
        if (isDc) {
            bitLength += 8;
            lastDc = lastDc + element.dc;
            decoded.push_back(static_cast<char>(lastDc));
            numBlocks++;
            isDc = false;
            continue;
        }
        if (element.EOB()) {
            bitLength += 2;
            decoded.resize(numBlocks * blockSize2, 0);
            isDc = true;
            continue;
        }
        // AC
        decoded.resize(decoded.size() + element.ac.run, 0);
        decoded.push_back(element.ac.value);
        bitLength += 14;
    }
    return {decoded, bitLength};
}


/*
 *  1. Обрезка:     uchar -> uchar
 *  2. Сдвиг:       uchar -> char
 *  3. ДКП:         char -> double
 *  4. Квантование: double -> char
 */

int main() {
    constexpr int blockSize = 8;
    constexpr int quality = 1;

    if (!std::filesystem::exists(outPath)) {
        std::filesystem::create_directory(outPath);
    }
    std::ofstream out((outPath / "out.txt").c_str(), std::ios::out);
    const cv::Mat image = cv::imread(R"(resources/1280pxEtu.jpg)", 0);
    const auto iw = image.cols / blockSize * blockSize, ih = image.rows / blockSize * blockSize;

    cv::Rect cropRect(0, 0, iw, ih);
    cv::Mat source(ih, iw, CV_8U);
    image(cropRect).copyTo(source);

    if (image.empty())
        return EXIT_FAILURE;

    const size_t pxCount = iw * ih;

    constexpr double H0{8.};
    // расчёты H, R
    {
        std::vector<uchar> pixels(source.data, source.data + pxCount);
        double H = imageEntropy(pixels);
        double R = 1 - H / H0;
        out << "=== SOURCE ===" << std::endl //
            << "H: " << H << std::endl       //
            << "H0: " << H0 << std::endl     //
            << "R: " << R << std::endl;
    }

    auto shifted = shift(source);                            // signed char
    auto dctCoefs = transformImage(shifted, blockSize);      // double
    auto quantised = quantise(dctCoefs, blockSize, quality); // signed char
    {
        std::vector<uchar> pixels(quantised.data, quantised.data + pxCount);
        double H = imageEntropy(pixels);
        double R = 1 - H / H0;
        out << "=== QUANTISED DCT ===" << std::endl //
            << "H: " << H << std::endl              //
            << "H0: " << H0 << std::endl            //
            << "R: " << R << std::endl;
    }
    std::vector<char> zigZagSource = zigZagRead(quantised, blockSize);
    std::vector<RleElement> rle = runLengthEncoding(zigZagSource, blockSize);
    auto [zigZagRecovered, bitsNum] = runLengthDecoding(rle, blockSize);
    {
        out << "=== RLE STREAM ===" << std::endl   //
            << "Length: " << bitsNum << std::endl  //
            << "Raw: " << 8 * pxCount << std::endl //
            << "K: " << 8 * static_cast<double>(pxCount) / static_cast<double>(bitsNum)
            << std::endl;
    }
    auto dctCoefsRecovered = zigZagWrite(zigZagRecovered, blockSize, iw, ih); // signed char
    auto dequantised = dequantise(dctCoefsRecovered, blockSize, quality);     // double
    auto shiftedRecovered = reverseTransformImage(dequantised, blockSize);    // signed char
    auto recovered = unshift(shiftedRecovered);                               // unsigned char
    cv::imshow("Source", source);
    cv::imshow("Recovered", recovered);
    cv::waitKey();
    return EXIT_SUCCESS;
}
