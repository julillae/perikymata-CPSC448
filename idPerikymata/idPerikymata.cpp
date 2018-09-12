#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat src;
Mat dest;

string stripName(string filename) {
    // Remove directory if present.
    // Do this before extension removal in case directory has a period character.
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

vector<int> measurePixelDistance(vector<Point> points) {
    vector<int> distances;
    for (int i = 1; i < points.size(); i++) {
        Point pt = points[i];
        Point last = points[i-1];
        int distance = pt.y - last.y;
        distances.push_back(distance);
    }

    return distances;
}

void filterPlateaus(vector<int> &dist, vector<Point> &troughs, vector<Point> &minima) {
    vector<Point> result;
    int i = 1;
    while (i < dist.size()) {
        int prev = troughs[i-1].y;
        int curr = troughs[i].y;
        if (std::abs(prev - curr) < 10 || dist[i-1] < 4 ) {
            troughs.erase(troughs.begin() + i);
            minima.erase(minima.begin() + i);
            dist.erase(dist.begin() + i - 1);
        } else {
            i++;
        }
    }
}

int writeDistancesCSV(vector<int> dists, string filename) {
    if (dists.size() > 0) {
        string file = filename + "_distances.csv";
        ofstream myfile;
        myfile.open(file.c_str());
        myfile << dists[0];
        for (int i = 1; i < dists.size(); i++) {
            myfile << "\n";
            myfile << dists[i];
        }
        myfile << endl;
        myfile.close();
    } else {
        return 0;
    }
}

int writePointssCSV(vector<Point> points, string filename) {
    if (points.size() > 0) {
        string file = filename + "_points.csv";
        ofstream myfile;
        myfile.open(file.c_str());
        myfile << "x,y";
        for (int i = 0; i < points.size(); i++) {
            myfile << ",\n";
            myfile << points[i].x;
            myfile << ",";
            myfile << points[i].y;
        }
        myfile << endl;
        myfile.close();
    } else {
        return 0;
    }
}

int localMaxima(Point pt, Mat img) {
    Mat cpy(img);
    int prev = int(cpy.at<uchar>(pt.y - 1, pt.x));
    int curr = int(cpy.at<uchar>(pt.y, pt.x));
    int next = int(cpy.at<uchar>(pt.y + 1, pt.x));
    if (curr > prev + 1 && curr > next + 1) {
        return 1;
    } else {
        return 0;
    }
}

int localMinimaFilter(Point pt, Mat img) {
    Mat cpy(img);
    int val = int(cpy.at<uchar>(pt.y, pt.x));
    double min;
    double max;
    Point min_pt(0,0), max_pt(0,0);
    Mat sub(cpy, Rect(pt.x, pt.y - 1, 1, 6));
    minMaxLoc(sub, &min, &max, &min_pt, &max_pt);
    if (std::abs(val - min) < 5 && std::abs(val - max) > 15) {
        return 1;
    } else {
        return 0;
    }
}

int isTrough(Point pt, Mat img) {
    Point left = Point(pt.x -1, pt.y);
    Point right = Point(pt.x + 1, pt.y);
    return localMinimaFilter(pt, img) && (localMinimaFilter(left, img) || localMinimaFilter(right, img));
}

int isTroughBasic(Point pt, Mat img) {
    Point left = Point(pt.x -1, pt.y);
    Point right = Point(pt.x + 1, pt.y);
    return localMaxima(pt, img) && (localMaxima(left, img) && localMaxima(right, img));
}

void transformKMeans(Mat org, Mat kmeans_image) {
    Mat img(org);
    Mat samples(img.rows * img.cols, 3, CV_32F);
    for( int y = 0; y < img.rows; y++ )
        for( int x = 0; x < img.cols; x++ )
        for( int z = 0; z < 3; z++)
            samples.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y,x)[z];

    int clusterCount = 10;
    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01), attempts, KMEANS_PP_CENTERS, centers );
    for( int y = 0; y < img.rows; y++ ) {
        for( int x = 0; x < img.cols; x++ ) { 
        int cluster_idx = labels.at<int>(y + x*img.rows,0);
        kmeans_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
        kmeans_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
        kmeans_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }
}

void printHelp() {
    printf(" Usage: ./histogram [image_name] [transect --default width/2] \n");
}

int main(int argc, char** argv) {
    namedWindow("Original", WINDOW_NORMAL);
   char* filename;
   int transect;
   if (argc >= 2) {
        filename = argv[1];
        src = imread( filename, IMREAD_COLOR );
        if (src.empty()) {
            printf(" Error opening image\n");
            printHelp();
            return -1;
        }
        transect = argc >=3 ? atoi(argv[2]) : src.cols/2;
        if (transect >= src.cols || transect <= 0) {
            printf(" Transect is out of bounds: the width of %s is %i \n", filename, src.cols);
            printHelp();
            return -1;
        }
   } else {
       printHelp();
       return -1;
   }    

    cvtColor( src, src, COLOR_BGR2GRAY );
    int i = 10;
    bilateralFilter ( src, dest, i, i*2, i/2 );

    int graph_w = dest.rows, graph_h = 260;
    Mat histImage( graph_h, graph_w, CV_32F, Scalar( 0,0,0) );
    vector<Point> minima;
    vector<Point> troughs;
    for( int i = 1; i < dest.rows-5; i++ ) {
        
        int currIntensity = int(dest.at<uchar>(i, transect));
        int prevIntensity = int(dest.at<uchar>(i - 1, transect));
        int nextIntensity = int(dest.at<uchar>(i + 1, transect));
        line( histImage, Point(i-1, graph_h - prevIntensity),
              Point(i, graph_h - currIntensity),
              Scalar( 255, 255, 255), 2, 8, 0  );
        
        Point curr = Point(transect, i);
        
        if (isTrough(curr, dest)) {
            minima.push_back(Point(i, graph_h - currIntensity));
            troughs.push_back(curr);
        }
    }
    vector<int> distances =  measurePixelDistance(troughs);
    cout << "Finished measuring pixel distance" << endl;
    filterPlateaus(distances, troughs, minima);
    cout << "Finished filtering plateaus" << endl;
    Mat peaks(histImage.rows, histImage.cols, CV_32FC3);
    Mat in_h[] = { histImage.clone(), histImage.clone(), histImage.clone() };
    int from_to_h[] = { 0,0, 1,1, 2,2 };
    mixChannels( in_h, 3, &peaks, 1, from_to_h, 3 );
    for ( int i = 0; i < minima.size(); i++) {
        drawMarker(dest, troughs[i], Scalar(0,0,255), MARKER_STAR, 5, 1);
        drawMarker(peaks, minima[i], Scalar(0,0,255), MARKER_STAR, 20, 1);
    }
    
    string basename = stripName(filename);

    writeDistancesCSV(distances, basename);
    writePointssCSV(troughs, basename);
    string pointsName = basename + "_points";
    string peaksName = basename + "_peaks";
    imwrite(pointsName + ".tif", dest);
    imwrite(peaksName + ".tif", peaks);
    cout << "Number of perikymata identified: " << minima.size() << endl;
    cout << "Finished!" << endl;
    return 0;
}