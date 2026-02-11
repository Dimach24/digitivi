#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <fstream>
int main() {
    std::ofstream outfile("resources/check.png");
    outfile << "P3\n" << 255 << '\n';
    return EXIT_SUCCESS;
}
