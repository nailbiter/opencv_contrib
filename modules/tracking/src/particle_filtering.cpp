#include "precomp.hpp"
#undef ALEX_DEBUG

#ifdef ALEX_DEBUG
#define dprintf(x) printf x;fflush(stdout)
    static void print_matrix(const cv::Mat& x){
        for(int i=0;i<x.rows;i++){
            printf("\t[");
            for(int j=0;j<x.cols;j++){
                printf("%g, ",x.at<double>(i,j));
            }
            printf("]\n");
        }
    }
#else
#define dprintf(x)
#define print_matrix(x)
#endif
#include "PFSolver.hpp"
#include "TrackingFunctionPF.hpp"

namespace cv{

    class TrackerTargetStatePF : public TrackerTargetState{
    public:
        Rect getRect(){return rect;}
        TrackerTargetStatePF(const Mat_<double>& row){
            CV_Assert(row.rows==1 && row.cols==4);
            rect=Rect(Point_<int>((int)row(0,0),(int)row(0,1)),Point_<int>((int)row(0,2),(int)row(0,3)));
        }
    protected:
        Rect rect;
    };

    class TrackerModelPF : public TrackerModel{
    public:
        TrackerModelPF( const TrackerPF::Params &parameters,const Mat& image,const Rect& boundingBox );
    protected:
        void modelEstimationImpl( const std::vector<Mat>& responses );
        void modelUpdateImpl();
    private:
        Ptr<PFSolver> _solver;
        Ptr<TrackingFunctionPF> _function;
        Mat_<double> _std,_last_guess;
    };

    void TrackerPF::Params::read(const FileNode& ){
    }
    void TrackerPF::Params::write( FileStorage& ) const{
    }
    void TrackerModelPF::modelUpdateImpl(){
    }

    TrackerModelPF::TrackerModelPF( const TrackerPF::Params &parameters,const Mat& image,const Rect& boundingBox ) :
        _function(new TrackingFunctionPF(image(boundingBox))){
        _solver=createPFSolver(_function,parameters.std,TermCriteria(TermCriteria::MAX_ITER,parameters.iterationNum,0.0),
                    parameters.particlesNum,parameters.alpha);
        _std=parameters.std;
        _last_guess=(Mat_<double>(1,4)<<(double)boundingBox.x,(double)boundingBox.y,
                (double)boundingBox.x+boundingBox.width,(double)boundingBox.y+boundingBox.height);
    }
    void TrackerModelPF::modelEstimationImpl( const std::vector<Mat>& responses ){
        CV_Assert(responses.size()==1);
        Mat image=responses[0];
        Ptr<TrackerTargetState> ptr;
        if(true){
            //TODO - here we do iterations
            _solver->setParamsSTD(_std);
            _solver->minimize(_last_guess);
            dynamic_cast<TrackingFunctionPF*>(static_cast<optim::Solver::Function*>(_solver->getFunction()))->update(image);
            while(_solver->iteration() <= _solver->getTermCriteria().maxCount);
            _solver->getOptParam(_last_guess);
            ptr=Ptr<TrackerTargetStatePF>(new TrackerTargetStatePF(_last_guess));
        }else{
            //Mat_<double> row=(Mat_<double>(1,4)<<0.0,0.0,(double)image.cols/2,(double)image.rows/2);
            Mat_<double> row=_last_guess;
            TrackerTargetStatePF *real_ptr=new TrackerTargetStatePF(row);
            ptr=Ptr<TrackerTargetState>(real_ptr);
            dynamic_cast<TrackingFunctionPF*>(static_cast<optim::Solver::Function*>(_solver->getFunction()))->update(image(real_ptr->getRect()));
        }

        dprintf(("before setLastTargetState() line %d\n",__LINE__));
        setLastTargetState(ptr);
        dprintf(("after setLastTargetState() line %d\n",__LINE__));
    }
    TrackerPF::Params::Params(){
        //if these def params will be changed, it might also have sense to change def params at optim.hpp
        iterationNum=20;
        particlesNum=100;
        alpha=0.9;
        std=(Mat_<double>(1,4)<<15.0,15.0,15.0,15.0); 
    }
    TrackerPF::TrackerPF( const TrackerPF::Params &parameters){
        params=parameters;
        isInit=false;
    }
    bool TrackerPF::initImpl( const Mat& image, const Rect& boundingBox ){
        model=Ptr<TrackerModel>(new TrackerModelPF(params,image,boundingBox));
        return true;
    }
    bool TrackerPF::updateImpl( const Mat& image, Rect& boundingBox ){
        model->modelEstimation(std::vector<Mat>(1,image));
        TrackerTargetStatePF* state=dynamic_cast<TrackerTargetStatePF*>(static_cast<TrackerTargetState*>(model->getLastTargetState()));
        boundingBox=state->getRect();
        return true;
    }
}
