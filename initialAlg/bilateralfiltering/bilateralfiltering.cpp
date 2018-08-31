#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

Mat src;
Mat gray;
Mat smooth;
Mat equalize1;
Mat equalize2;
const int i = 15;

string stripName(string filename);

int main( int argc, char ** argv ) {
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Equalized1", WINDOW_NORMAL);
    namedWindow("Smoothed", WINDOW_NORMAL);
    namedWindow("Equalized2", WINDOW_NORMAL);
    const char* filename = argc >=2 ? argv[1] : "../data/ALB51-LLC-13.tif";
    src = imread( filename, IMREAD_COLOR );
    if (src.empty()) {
            printf(" Error opening image\n");
            printf(" Usage: ./smoothing [image_name -- default ../data/ALB51-LLC-13.tif] \n");
            return -1;
        }

    cvtColor( src, gray, COLOR_BGR2GRAY );

    equalizeHist(gray, equalize1);
    string equalized1Filename = stripName(filename) + "_equalized1";
    imwrite("../data/" + equalized1Filename + ".tif", equalize1);

    bilateralFilter ( gray, smooth, i, i*2, i/2 );
    string smoothedFilename = stripName(filename) + "_smoothed";
    imwrite("../data/" + smoothedFilename +".tif", smooth);
    
    equalizeHist(smooth, equalize2);
    string equalized2Filename = stripName(filename) + "_equalized2";
    imwrite("../data/" + equalized2Filename + ".tif", equalize2);

    imshow("Original", src);
    imshow("Smoothed", smooth);
    imshow("Equalized1", equalize1);
    imshow("Equalized2", equalize2);
    waitKey(0);
}

string stripName(string filename) {
    // Remove directory if present.
    // Do this before extension removal incase directory has a period character.
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        filename.erase(0, last_slash_idx + 1);
    }

    // Remove extension if present.
    const size_t period_idx = filename.rfind('.');
    if (std::string::npos != period_idx)
    {
        filename.erase(period_idx);
    }
    
    return filename;
}