#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

string stripName(string filename);

Mat src;

int main( int argc, char** argv ) {
    const char* filename = argc >=2 ? argv[1] : "../data/ALB51-LLC-13.tif";
    src = imread( filename, IMREAD_COLOR );
    Mat samples(src.rows * src.cols, 3, CV_32F);
    for( int y = 0; y < src.rows; y++ )
        for( int x = 0; x < src.cols; x++ )
        for( int z = 0; z < 3; z++)
            samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


    int clusterCount = 6;
    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01), attempts, KMEANS_PP_CENTERS, centers );


    Mat new_image( src.size(), src.type() );
    for( int y = 0; y < src.rows; y++ ) {
        for( int x = 0; x < src.cols; x++ ) { 
        int cluster_idx = labels.at<int>(y + x*src.rows,0);
        new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
        new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
        new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }

    string kmeansFilename = stripName(filename) + "_kmeans";
    imwrite("../data/" + kmeansFilename + ".tif", new_image);
    namedWindow("clustered image", WINDOW_NORMAL);
    imshow( "clustered image", new_image );
    waitKey( 0 );
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
    if (std::string::npos != period_idx) {
        filename.erase(period_idx);
    }
    
    return filename;
}
