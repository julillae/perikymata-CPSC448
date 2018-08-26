#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat src;
Mat rbg;

string stripName(string filename) {
    // Remove directory if present.
    // Do this before extension removal incase directory has a period character.
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
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

vector<int> measurePixelDistance(vector<Point> points) {
    vector<int> distances;
    for (int i = 1; i < points.size(); i++) {
        Point pt = points[i];
       // cout << "Point is " << pt.x << " and " << pt.y << endl;
        Point last = points[i-1];
       // cout << "Last is " << pt.x << " and " << pt.y << endl;
        int distance = pt.y - last.y;
        distances.push_back(distance);
      //  cout << "Added " << distance << " distance" << endl;
    }
    return distances;
}

void filterPlateaus(vector<int> &dist, vector<Point> &troughs, vector<Point> &minima) {
    vector<Point> result;
    int i = 1;
    while (i < troughs.size()) {
        int prev = troughs[i-1].y;
        int curr = troughs[i].y;
        if (std::abs(prev - curr) < 10) {
            troughs.erase(troughs.begin() + i);
            minima.erase(minima.begin() + i);
            dist.erase(dist.begin() + i - 1);
        } else {
            i++;
        }
    }
}

int writeCSVFile(vector<int> dists, string filename) {
    if (dists.size() > 0) {
        const char* file = "distances.csv";
        ofstream myfile;
        myfile.open(file);
        myfile << dists[0];
        for (int i = 1; i < dists.size(); i++) {
            myfile << ",";
            myfile << dists[i];
        }
        myfile << endl;
        myfile.close();
    } else {
        return 0;
    }
}

int localMaxima(Point pt, Mat img) {
    Mat cpy(img);
    cout << "localMaxima point is " << pt.x << " and " << pt.y << endl;
    int prev = int(cpy.at<uchar>(pt.y - 1, pt.x));
    int curr = int(cpy.at<uchar>(pt.y, pt.x));
    cout << "Passed intensity is " << curr << endl;
    int next = int(cpy.at<uchar>(pt.y + 1, pt.x));
    if (curr > prev + 1 && curr > next + 1) {
        return 1;
    } else {
        return 0;
    }
}

int localMinimaFilter(Point pt, Mat img) {
    Mat cpy(img);
    // int prev = int(cpy.at<uchar>(pt.y - 3, pt.x));
    // int next = int(cpy.at<uchar>(pt.y + 3, pt.x));
    int val = int(cpy.at<uchar>(pt.y, pt.x));
    double min;
    double max;
    Point min_pt(0,0), max_pt(0,0);
    Mat sub(cpy, Rect(pt.x, pt.y - 1, 1, 6));
    minMaxLoc(sub, &min, &max, &min_pt, &max_pt);
    if (std::abs(val - min) < 10 && std::abs(val - max) > 25) {
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
    // cout << "isTrough Point is " << pt.x << " and " << pt.y << endl;
    Point left = Point(pt.x -1, pt.y);
    Point right = Point(pt.x + 1, pt.y);
    return localMaxima(pt, img) && (localMaxima(left, img) || localMaxima(right, img));
}


int main(int argc, char** argv) {
    namedWindow("Original", WINDOW_NORMAL);
    // namedWindow("Hist", WINDOW_NORMAL);
    // namedWindow("Gray", WINDOW_NORMAL);
   //  namedWindow("Peaks", WINDOW_NORMAL);
    const char* filename = argc >=2 ? argv[1] : "../data/ALB51-LLC-13.tif";
    src = imread( filename, IMREAD_COLOR );
    if (src.empty()) {
            printf(" Error opening image\n");
            printf(" Usage: ./histogram [image_name -- default ../data/ALB51-LLC-13.tif] \n");
            return -1;
        }

    cvtColor( src, src, COLOR_BGR2GRAY );
    int i = 5;
    bilateralFilter ( src, rbg, i, i*2, i/2 );
    // GaussianBlur(rbg, rbg, Size(3,3), 0);

    bool uniform = true, accumulate = false;
    int graph_w = rbg.rows, graph_h = 260;
    Mat histImage( graph_h, graph_w, CV_32F, Scalar( 0,0,0) );
    vector<Point> minima;
    vector<Point> troughs;
    int transect = rbg.cols/2;
    for( int i = 1; i < rbg.rows-5; i++ )
    {
        
        int currIntensity = int(rbg.at<uchar>(i, transect));
        int prevIntensity = int(rbg.at<uchar>(i - 1, transect));
        int nextIntensity = int(rbg.at<uchar>(i + 1, transect));
        line( histImage, Point(i-1, graph_h - prevIntensity),
              Point(i, graph_h - currIntensity),
              Scalar( 255, 255, 255), 2, 8, 0  );
        
        // cout << "Intensity is " << currIntensity << endl;
        Point curr = Point(transect, i);
        
        if (isTrough(curr, rbg)) {
            minima.push_back(Point(i, graph_h - currIntensity));
            troughs.push_back(curr);
            cout << "Point is " << curr.x << " and " << curr.y << endl;
        }
    }
    cout << "Finished iteration" << endl;
    vector<int> distances =  measurePixelDistance(troughs);
    cout << "Finished measuring pixel distance" << endl;
    filterPlateaus(distances, troughs, minima);
    cout << "Finished filtering plateaus" << endl;
    Mat peaks(histImage.rows, histImage.cols, CV_32FC3);
    Mat in_h[] = { histImage.clone(), histImage.clone(), histImage.clone() };
    int from_to_h[] = { 0,0, 1,1, 2,2 };
    mixChannels( in_h, 3, &peaks, 1, from_to_h, 3 );

    for ( int i = 0; i < minima.size(); i++) {
        drawMarker(rbg, troughs[i], Scalar(0,0,255), MARKER_STAR, 5, 1);
        drawMarker(peaks, minima[i], Scalar(0,0,255), MARKER_STAR, 20, 1);
    }

    cout << "Distance vector length: " << distances.size() << endl;
    writeCSVFile(distances, filename);
    
    string peaksName = stripName(filename) + "_peaks";
    imwrite("../data/" + peaksName + ".tif", peaks);
    cout << minima.size() << endl;
    imshow("Original", src );
    imshow("Hist", histImage );
    imshow("Gray", rbg);
    imshow("Peaks", peaks);
    waitKey(0);
    return 0;
}