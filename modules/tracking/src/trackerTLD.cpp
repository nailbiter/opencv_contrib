/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "time.h"
#include <algorithm>
#include <limits.h>
#include "TLD.hpp"

#define HOW_MANY_CLASSIFIERS 20
#define THETA_NN 0.5

using namespace cv;

/*
 * FIXME(optimize):
 *
 *   skeleton!!!
*/
/*ask Kalal: 
 * ./bin/example_tracking_tracker TLD ../TrackerChallenge/test.avi 0 5,110,25,130 > out.txt
 *
 *  init_model:negative_patches  -- all?
 *  posterior: 0/0
 *  sampling: how many base classifiers?
 *  initial model: why 20
 *  scanGrid low overlap
 *  rotated rect in initial model
 */

namespace cv
{


class TrackerTLDModel;

class TLDDetector : public TrackerTLD::Private{
public:
    TLDDetector(const TrackerTLD::Params& params,int rows,int cols,Rect2d initBox);
    ~TLDDetector(){}
    std::vector<Rect2d>& generateScanGrid(){return scanGrid;}
    void setModel(Ptr<TrackerModel> model_in){model=model_in;}
    Rect2d detect(const Mat& img,const Mat& imgBlurred,bool& hasFailed_out);
    bool getNextRect(Rect2d& rect,bool& isObjectFlag,bool reset=false);
protected:
    int patchVariance(const Mat& img,double originalVariance,int size);
    int ensembleClassifier(const Mat& blurredImg,std::vector<TLDEnsembleClassifier>& classifiers,int size);
    std::vector<Rect2d> scanGrid;
    TrackerTLD::Params params_;
    Ptr<TrackerModel> model;
    int scanGridPos;
    std::vector<bool> isObject;
};

void Pexpert(Rect2d trackerBox,TrackerTLDModel* model,Mat* img,Mat* imgBlurred);

void Nexpert(Rect2d trackerBox,TrackerTLDModel* model,Mat* img,Mat* imgBlurred,Rect2d box);

template <class T,class Tparams>
class TrackerProxyImpl : public TrackerProxy{
public:
    TrackerProxyImpl(Tparams params=Tparams()):params_(params),isConfident_(false){}
    bool init( const Mat& image, const Rect2d& boundingBox ){
        isConfident_=false;
        trackerPtr=Ptr<T>(new T(params_));
        trackerPtr->init(image,boundingBox);
        boundingBox_=boundingBox_;
        return true;
    }
    Rect2d update( const Mat& image,bool hasFailed_out){
        if(false){
            hasFailed_out=!trackerPtr->update(image,boundingBox_);
            isConfident_=isConfident_ && hasFailed_out;
            return boundingBox_;
        }else{
            hasFailed_out=true;
            return Rect2d(0,0,0,0);
        }
    }
    void setConfident(){isConfident_=true;}
    bool isConfident(){return isConfident_;}
private:
    Ptr<T> trackerPtr;
    bool isConfident_;
    Tparams params_;
    Rect2d boundingBox_;
};

class TrackerTLDModel : public TrackerModel{
 public:
  TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox,TLDDetector* detector);
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  double getOriginalVariance(){return originalVariance_;}
  std::vector<TLDEnsembleClassifier>* getClassifiers(){return &classifiers;}
  double Sr(const Mat_<double> patch);
  double Sc(const Mat_<double> patch);
 protected:
  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}
  Rect2d boundingBox_;
  double originalVariance_;
  std::vector<Mat_<double> > positiveExamples,negativeExamples;
  RNG rng;
  std::vector<TLDEnsembleClassifier> classifiers;
};


TrackerTLD::Params::Params(){
}

void TrackerTLD::Params::read( const cv::FileNode& fn ){
}

void TrackerTLD::Params::write( cv::FileStorage& fs ) const{
}

TrackerTLD::TrackerTLD( const TrackerTLD::Params &parameters) :
    params( parameters ){
  isInit = false;
  privateInfo.push_back(Ptr<TrackerProxyImpl<TrackerMedianFlow,TrackerMedianFlow::Params> >(
              new TrackerProxyImpl<TrackerMedianFlow,TrackerMedianFlow::Params>()));
}

TrackerTLD::~TrackerTLD(){
}

void TrackerTLD::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerTLD::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

bool TrackerTLD::initImpl(const Mat& image, const Rect2d& boundingBox ){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    TLDDetector* detector=new TLDDetector(params,image.rows,image.cols,boundingBox);
    privateInfo.push_back(Ptr<TLDDetector>(detector));
    model=Ptr<TrackerTLDModel>(new TrackerTLDModel(params,image_gray,boundingBox,detector));
    detector->setModel(model);
    ((TrackerProxy*)static_cast<Private*>(privateInfo[0]))->init(image,boundingBox);
    return true;
}

bool TrackerTLD::updateImpl(const Mat& image, Rect2d& boundingBox){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    Mat image_blurred;
    GaussianBlur(image_gray,image_blurred,Size(3,3),0.0);
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    TLDDetector* detector=((TLDDetector*)static_cast<TrackerTLD::Private*>(privateInfo[1]));
    TrackerProxy* trackerProxy=(TrackerProxy*)static_cast<Private*>(privateInfo[0]);

    //best overlap around 92%
    /*double m=0;
    for(int i=0;i<scanGrid.size();i++){
        double overlap=TLDDetector::overlap(scanGrid[i],boundingBox);
        if(overlap>m){m=overlap;}
    }
    printf("best overlap: %f\n",m);*/

    bool detectorFailed=true,trackerFailed=true,needToRestartTracker=false;
    Rect2d detectorAnswer=detector->detect(image_gray,image_blurred,detectorFailed),
           trackerAnswer=trackerProxy->update(image,trackerFailed);

    //TODO: decide best box and whether we'll need to restart the tracker
    if(detectorFailed && trackerFailed){
        return false;
    }
    if(trackerFailed){
        boundingBox=detectorAnswer;
    }

    if(trackerProxy->isConfident()){
        //TODO: increase tracker's confidence
        //TODO: P expert
        //TODO: N expert
    }

    if(needToRestartTracker){
        trackerProxy->init(image,boundingBox);
    }
    return true;
}

TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox,TLDDetector* detector){
    boundingBox_=boundingBox;
    originalVariance_=variance(image(boundingBox));
    std::vector<Rect2d> scanGrid=detector->generateScanGrid(),closest(10);

    getClosestN(scanGrid,boundingBox,10,closest);

    Mat image_blurred;
    Mat_<double> blurredPatch(15,15);
    GaussianBlur(image,image_blurred,Size(3,3),0.0);
    for(int i=0;i<HOW_MANY_CLASSIFIERS;i++){
        classifiers.push_back(TLDEnsembleClassifier(i+1));
    }

    positiveExamples.reserve(200);
    Point2f center;
    Size2f size;
    for(int i=0;i<closest.size();i++){
        for(int j=0;j<20;j++){
            Mat_<double> standardPatch(15,15);
            center.x=closest[i].x+closest[i].width*(0.5+rng.uniform(-0.01,0.01));
            center.y=closest[i].y+closest[i].height*(0.5+rng.uniform(-0.01,0.01));
            size.width=closest[i].width*rng.uniform((double)0.99,(double)1.01);
            size.height=closest[i].height*rng.uniform((double)0.99,(double)1.01);
            float angle=rng.uniform((double)-10.0,(double)10.0);

            resample(image,RotatedRect(center,size,angle),standardPatch);
            for(int y=0;y<standardPatch.rows;y++){
                for(int x=0;x<standardPatch.cols;x++){
                    standardPatch(x,y)+=rng.gaussian(5.0);
                }
            }
            positiveExamples.push_back(standardPatch);

            resample(image_blurred,RotatedRect(center,size,angle),blurredPatch);
            for(int k=0;k<classifiers.size();k++){
                classifiers[k].integrate(blurredPatch,true);
            }
        }
    }

    negativeExamples.clear();
    const int negMax=200;
    negativeExamples.reserve(negMax);
    std::vector<int> indices;
    indices.reserve(negMax);
    while(negativeExamples.size()<negMax){
        int i=rng.uniform((int)0,(int)scanGrid.size());
        if(std::find(indices.begin(),indices.end(),i)==indices.end() && overlap(boundingBox,scanGrid[i])<0.2){
            Mat_<double> standardPatch(15,15);
            resample(image,scanGrid[i],standardPatch);
            negativeExamples.push_back(standardPatch);

            resample(image_blurred,scanGrid[i],blurredPatch);
            for(int k=0;k<classifiers.size();k++){
                classifiers[k].integrate(blurredPatch,false);
            }
        }
    }
    for(int i=rng.uniform((int)0,(int)negativeExamples.size());negativeExamples.size()>400;i=rng.uniform((int)0,(int)negativeExamples.size())){
        negativeExamples.erase(negativeExamples.begin()+i);
    }
    printf("positive patches: %d\nnegative patches: %d\n",positiveExamples.size(),negativeExamples.size());
}


TLDDetector::TLDDetector(const TrackerTLD::Params& params,int rows,int cols,Rect2d initBox){
    scanGrid.clear();
    //scales step: 1.2; hor step: 10% of width; verstep: 10% of height; minsize: 20pix
    for(double h=initBox.height, w=initBox.width;h<cols && w<rows;){
        for(double x=0;(x+w)<cols;x+=(0.1*w)){
            for(double y=0;(y+h)<rows;y+=(0.1*h)){
                scanGrid.push_back(Rect2d(x,y,w,h));
            }
        }
        if(h<=initBox.height){
            h/=1.2; w/=1.2;
            if(h<20 || w<20){
                h=initBox.height*1.2; w=initBox.width*1.2;
                CV_Assert(h>initBox.height || w>initBox.width);
            }
        }else{
            h*=1.2; w*=1.2;
        }
    }
    printf("%d rects in scanGrid\n",scanGrid.size());
}

Rect2d TLDDetector::detect(const Mat& img,const Mat& imgBlurred,bool& hasFailed_out){
    int remains=0;
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));

    MEASURE_TIME(remains=patchVariance(img,tldModel->getOriginalVariance(),scanGrid.size());)
    printf("remains %d rects\n",remains);

    MEASURE_TIME(remains=ensembleClassifier(imgBlurred,*(tldModel->getClassifiers()),remains);)
    printf("remains %d rects\n",remains);

    Mat_<double> standardPatch(15,15);
    float maxSc=0.0;
    Rect2d maxScRect,tmpScRect;
    double tmpSc=0.0;
    int iSc=-1;
    isObject.resize(remains);
    for(int i=0;i<remains;i++){
        tmpScRect=scanGrid[i];
        resample(img,tmpScRect,standardPatch);
        isObject[i]=((tldModel->Sr(standardPatch))>THETA_NN);
        if(!isObject[i]){
            continue;
        }
        tmpSc=tldModel->Sc(standardPatch);
        if(tmpSc>maxSc){
            iSc=i;
            maxSc=tmpSc;
            maxScRect=tmpScRect;
        }
    }
    if(iSc==-1){
        hasFailed_out=true;
    }
    printf("iSc=%d\n",iSc);

    hasFailed_out=false;
    return maxScRect;
}

bool TLDDetector::getNextRect(Rect2d& rect,bool& isObjectFlag,bool reset){
    if(reset){
        scanGridPos=0;
    }
    if(scanGridPos>=isObject.size()){
        return false;
    }
    rect=scanGrid[scanGridPos];
    isObjectFlag=isObject[scanGridPos];
    scanGridPos++;
    return true;
}

int TLDDetector::patchVariance(const Mat& img,double originalVariance,int size){
    Mat_<unsigned int> intImgP(img.rows,img.cols),intImgP2(img.rows,img.cols);

    intImgP(0,0)=img.at<uchar>(0,0);
    for(int j=1;j<intImgP.cols;j++){intImgP(0,j)=intImgP(0,j-1)+img.at<uchar>(0,j);}
    for(int i=1;i<intImgP.rows;i++){intImgP(i,0)=intImgP(i-1,0)+img.at<uchar>(i,0);}
    for(int i=1;i<intImgP.rows;i++){for(int j=1;j<intImgP.cols;j++){
            intImgP(i,j)=intImgP(i,j-1)-intImgP(i-1,j-1)+intImgP(i-1,j)+img.at<uchar>(i,j);}}

    unsigned int p;
    p=img.at<uchar>(0,0);intImgP2(0,0)=p*p;
    for(int j=1;j<intImgP2.cols;j++){p=img.at<uchar>(0,j);intImgP2(0,j)=intImgP2(0,j-1)+p*p;}
    for(int i=1;i<intImgP2.rows;i++){p=img.at<uchar>(i,0);intImgP2(i,0)=intImgP2(i-1,0)+p*p;}
    for(int i=1;i<intImgP2.rows;i++){for(int j=1;j<intImgP2.cols;j++){p=img.at<uchar>(i,j);
            intImgP2(i,j)=intImgP2(i,j-1)-intImgP2(i-1,j-1)+intImgP2(i-1,j)+p*p;}}

    int i=0,j=0;
    Rect2d tmp;

    for(;i<size && !(variance(intImgP,intImgP2,img,(scanGrid[i]))<0.5*originalVariance);i++);

    for(j=i+1;j<size;j++){
        if(!(variance(intImgP,intImgP2,img,(scanGrid[j]))<0.5*originalVariance)){
            tmp=scanGrid[i];
            scanGrid[i]=scanGrid[j];
            scanGrid[j]=tmp;
            i++;
        }
    }

    return MIN(size,i+1);
}

int TLDDetector::ensembleClassifier(const Mat& blurredImg,std::vector<TLDEnsembleClassifier>& classifiers,int size){
    Mat_<double> standardPatch(15,15);
    int i=0,j=0;
    Rect2d tmp;

    for(;i<size;i++){
        double p=0.0;
        resample(blurredImg,scanGrid[i],standardPatch);
        for(int k=0;k<classifiers.size();k++){
            p+=classifiers[k].posteriorProbability(standardPatch);
        }
        p/=classifiers.size();

        if(p<=0.5){
            break;
        }
    }

    for(j=i+1;j<size;j++){
        double p=0.0;
        resample(blurredImg,scanGrid[j],standardPatch);
        for(int k=0;k<classifiers.size();k++){
            p+=classifiers[k].posteriorProbability(standardPatch);
        }
        p/=classifiers.size();

        if(!(p<=0.5)){
            tmp=scanGrid[i];
            scanGrid[i]=scanGrid[j];
            scanGrid[j]=tmp;
            i++;
        }
    }

    return MIN(size,i+1);
}

double TrackerTLDModel::Sr(const Mat_<double> patch){
    double splus=0.0;
    for(int i=0;i<positiveExamples.size();i++){
        splus=MAX(splus,0.5*(NCC(positiveExamples[i],patch)+1.0));
    }
    double sminus=0.0;
    for(int i=0;i<negativeExamples.size();i++){
        sminus=MAX(sminus,0.5*(NCC(negativeExamples[i],patch)+1.0));
    }
    if(splus+sminus==0.0){
        return 0.0;
    }
    return splus/(sminus+splus);
}

double TrackerTLDModel::Sc(const Mat_<double> patch){
    double splus=0.0;
    for(int i=0;i<((positiveExamples.size()+1)/2);i++){
        splus=MAX(splus,0.5*(NCC(positiveExamples[i],patch)+1.0));
    }
    double sminus=0.0;
    for(int i=0;i<negativeExamples.size();i++){
        sminus=MAX(sminus,0.5*(NCC(negativeExamples[i],patch)+1.0));
    }
    if(splus+sminus==0.0){
        return 0.0;
    }
    return splus/(sminus+splus);
}

void Pexpert(Rect2d trackerBox,TrackerTLDModel* model,Mat* img,Mat* imgBlurred){
    //TODO
}

void Nexpert(Rect2d trackerBox,TrackerTLDModel* model,Mat* img,Mat* imgBlurred,Rect2d box){
    //TODO
}

} /* namespace cv */
