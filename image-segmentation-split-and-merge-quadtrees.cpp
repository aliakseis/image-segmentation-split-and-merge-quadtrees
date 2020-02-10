// image-segmentation-split-and-merge-quadtrees.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

//Mat img;
//Size size;

struct region {
    // tree data structure
    vector<region> childs;
    bool validity; // TODO: have a method for clear the data structure and remove regions with false validity

    // tree for split&merge procedure
    Rect roi;
    Mat m;
    Scalar label;
    //Mat mask; // for debug. don't use in real cases because it is computationally too heavy.
};

//----------------------------------------------------------------------------------------------------------------------- merging
bool mergeTwoRegion(region& parent, const Mat& src, region& r1, region& r2, bool(*predicate)(const Mat&)) {
    if (r1.childs.size() == 0 && r2.childs.size() == 0) {

        Rect roi1 = r1.roi;
        Rect roi2 = r2.roi;
        Rect roi12 = roi1 | roi2;
        if (predicate(src(roi12))) {
            r1.roi = roi12;

            // recompute mask
            //r1.mask = Mat::zeros(size, CV_8U);
            //rectangle(r1.mask, r1.roi, 1, CV_FILLED);

            r2.validity = false;
            return true;
        }
    }
    return false;
}

void merge(const Mat& src, region& r, bool(*predicate)(const Mat&)) {
    // check for adjiacent regions. if predicate is true, then  merge.
    // the problem is to check for adjiacent regions.. one way can be:
    // check merging for rows. if neither rows can be merged.. check for cols.

    bool row1 = false, row2 = false, col1 = false, col2 = false;

    if (r.childs.size() < 1) return;

    // try with the row
    row1 = mergeTwoRegion(r, src, r.childs[0], r.childs[1], predicate);
    row2 = mergeTwoRegion(r, src, r.childs[2], r.childs[3], predicate);

    if (!(row1 | row2)) {
        // try with column
        col1 = mergeTwoRegion(r, src, r.childs[0], r.childs[2], predicate);
        col2 = mergeTwoRegion(r, src, r.childs[1], r.childs[3], predicate);
    }

    for (int i = 0; i < r.childs.size(); i++) {
        if (r.childs[i].childs.size() > 0)
            merge(src, r.childs[i], predicate);
    }
}

//----------------------------------------------------------------------------------------------------------------------- quadtree splitting
region split(const Mat& src, Rect roi, bool(*predicate)(const Mat&)) {
    vector<region> childs;
    region r;

    r.roi = roi;
    r.m = src;
    //r.mask = Mat::zeros(size, CV_8U);
    //rectangle(r.mask, r.roi, 1, CV_FILLED);
    r.validity = true;

    bool b = predicate(src);
    if (b) {
        Scalar mean, s;
        meanStdDev(src, mean, s);
        r.label = mean;
    }
    else {
        int w = src.cols / 2;
        int h = src.rows / 2;
        region r1 = split(src(Rect(0, 0, w, h)), Rect(roi.x, roi.y, w, h), predicate);
        region r2 = split(src(Rect(w, 0, w, h)), Rect(roi.x + w, roi.y, w, h), predicate);
        region r3 = split(src(Rect(0, h, w, h)), Rect(roi.x, roi.y + h, w, h), predicate);
        region r4 = split(src(Rect(w, h, w, h)), Rect(roi.x + w, roi.y + h, w, h), predicate);
        r.childs.push_back(r1);
        r.childs.push_back(r2);
        r.childs.push_back(r3);
        r.childs.push_back(r4);
    }
    //merge(img, r, predicate);
    return r;
}

//----------------------------------------------------------------------------------------------------------------------- tree traversing utility
void print_region(region r) {
    if (r.validity == true && r.childs.size() == 0) {
        //cout << r.mask << " at ";
        cout << r.roi.x << "-" << r.roi.y << endl;
        cout << r.childs.size() << endl;
        cout << "---" << endl;
    }
    for (int i = 0; i < r.childs.size(); i++) {
        print_region(r.childs[i]);
    }
}

void draw_rect(Mat& imgRect, region r) {
    if (r.validity == true && r.childs.size() == 0)
        rectangle(imgRect, r.roi, 50, .1);
    for (int i = 0; i < r.childs.size(); i++) {
        draw_rect(imgRect, r.childs[i]);
    }
}

void draw_region(Mat& img, region r) {
    if (r.validity == true && r.childs.size() == 0)
        rectangle(img, r.roi, r.label, FILLED);
    for (int i = 0; i < r.childs.size(); i++) {
        draw_region(img, r.childs[i]);
    }
}

//----------------------------------------------------------------------------------------------------------------------- split&merge test predicates
bool predicateStdZero(const Mat& src) {
    Scalar stddev, mean;
    meanStdDev(src, mean, stddev);
    return stddev[0] == 0;
}
bool predicateStd5(const Mat& src) {
    Scalar stddev, mean;
    meanStdDev(src, mean, stddev);
    return (stddev[0] <= 5.8) || (src.rows*src.cols <= 25);
}

//----------------------------------------------------------------------------------------------------------------------- main
int main(int /*argc*/, char** /*argv*/)
{
    //img = (Mat_<uchar>(4, 4) << 0, 0, 1, 1,
    //    1, 1, 1, 1,
    //    3, 3, 3, 3,
    //    3, 4, 4, 3);

    //cout << img << endl;
    //size = img.size();

    //region r;
    //r = split(img, Rect(0, 0, img.cols, img.rows), &predicateStdZero);
    //merge(img, r, &predicateStdZero);
    //cout << "------- print" << endl;
    //print_region(r);

    cout << "-----------------------" << endl;

    auto img = imread(samples::findFile("lena.jpg"), 0);

    // round (down) to the nearest power of 2 .. quadtree dimension is a pow of 2.
    int exponent = log(min(img.cols, img.rows)) / log(2);
    int s = pow(2.0, (double)exponent);
    Rect square = Rect(0, 0, s, s);
    img = img(square).clone();

    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", img);

    cout << "now try to split.." << endl;
    auto r = split(img, Rect(0, 0, img.cols, img.rows), predicateStd5);

    cout << "splitted" << endl;
    Mat imgRect = img.clone();
    draw_rect(imgRect, r);
    namedWindow("split", WINDOW_AUTOSIZE);
    imshow("split", imgRect);
    imwrite("split.jpg", imgRect);

    merge(img, r, &predicateStd5);
    Mat imgMerge = img.clone();
    draw_rect(imgMerge, r);
    namedWindow("merge", WINDOW_AUTOSIZE);
    imshow("merge", imgMerge);
    imwrite("merge.jpg", imgMerge);

    Mat imgSegmented = img.clone();
    draw_region(imgSegmented, r);
    namedWindow("segmented", WINDOW_AUTOSIZE);
    imshow("segmented", imgSegmented);
    imwrite("segmented.jpg", imgSegmented);

    while (true)
    {
        char c = (char)waitKey(10);
        if (c == 27) { break; }
    }

    return 0;
}