#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

static Mat image;
static Rect2d boundingBox;
static bool paused;

static const char* keys =
{ "{@tracker_algorithm | | Tracker algorithm }"
    "{@video_name      | | video name        }"
    "{@start_frame     |0| Start frame       }" 
    "{@bounding_frame  |0,0,0,0| Initial bounding frame}"};

static void listTrackers(){
  vector<String> algorithms;
  Algorithm::getList(algorithms);
  cout << "\nAvailable tracker algorithms:\n";
  for (size_t i=0; i < algorithms.size(); i++){
      const char* algoname=algorithms[i].c_str();
      char *pos=NULL;
      if((pos=strstr((char*)algoname,"TRACKER."))!=NULL){
          printf("%s\n",pos+8);
      }
  }
}

static void help()
{
  cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
       "-- pause video [p] and draw a bounding box around the target to start the tracker\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./tracker <tracker_algorithm> <video_name> <start_frame> [<bounding_frame>]\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
  listTrackers();
}

int main( int argc, char** argv ){
  CommandLineParser parser( argc, argv, keys );

  String tracker_algorithm = parser.get<String>( 0 );
  String video_name = parser.get<String>( 1 );
  int start_frame = parser.get<int>( 2 );

  if( tracker_algorithm.empty() || video_name.empty() )
  {
    help();
    return -1;
  }

  int coords[4]={0,0,0,0};
  bool initBoxWasGivenInCommandLine=false;
  do{
      String initBoundingBox=parser.get<String>(3);
      for(size_t npos=0,pos=0,ctr=0;ctr<4;ctr++){
        npos=initBoundingBox.find_first_of(',',pos);
        if(npos==string::npos && ctr<3){
           printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer cordinates of opposed corners of bdd box\n");
           printf("got: %s\n",initBoundingBox.substr(pos,string::npos).c_str());
           printf("manual selection of bounding box will be employed\n");
           break;
        }
        int num=atoi(initBoundingBox.substr(pos,(ctr==3)?(string::npos):(npos-pos)).c_str());
        if(num<=0){
           printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer cordinates of opposed corners of bdd box\n");
           printf("got: %s\n",initBoundingBox.substr(pos,npos-pos).c_str());
           printf("manual selection of bounding box will be employed\n");
           break;
        }
        coords[ctr]=num;
        pos=npos+1;
      }
      if(coords[0]>0 && coords[1]>0 && coords[2]>0 && coords[3]>0){
          initBoxWasGivenInCommandLine=true;
      }
  }while(0);

  //open the capture
  VideoCapture cap;
  cap.open( video_name );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  if( !cap.isOpened() )
  {
    help();
    cout << "***Could not initialize capturing...***\n";
    cout << "Current parameter's value: \n";
    parser.printMessage();
    return -1;
  }

  Mat frame;
  paused = true;
  namedWindow( "Tracking API", 1 );
  setMouseCallback( "Tracking API", onMouse, 0 );

  //instantiates the specific Tracker
  Ptr<Tracker> tracker = Tracker::create( tracker_algorithm );
  if( tracker == NULL )
  {
    cout << "***Error in the instantiation of the tracker...***\n";
    return -1;
  }

  //get the first frame
  cap >> frame;
  frame.copyTo( image );
  if(initBoxWasGivenInCommandLine){
      selectObject=true;
      paused=false;
      boundingBox.x = coords[0];
      boundingBox.y = coords[1];
      boundingBox.width = std::abs( coords[2] - coords[0] );
      boundingBox.height = std::abs( coords[3]-coords[1]);
      printf("bounding box with vertices (%d,%d) and (%d,%d) was given in command line\n",coords[0],coords[1],coords[2],coords[3]);
      rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
  }
  imshow( "Tracking API", image );

  bool initialized = false;
  int frameCounter = 0;

  for ( ;; )
  {
    if( !paused )
    {
      if(initialized){
          cap >> frame;
          if(frame.empty()){
            break;
          }
          frame.copyTo( image );
      }

      if( !initialized && selectObject )
      {
        //initializes the tracker
        if( !tracker->init( frame, boundingBox ) )
        {
          cout << "***Could not initialize tracker...***\n";
          return -1;
        }
        initialized = true;
      }
      else if( initialized )
      {
        //updates the tracker
        if( tracker->update( frame, boundingBox ) )
        {
          rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
        }
      }
      imshow( "Tracking API", image );
      frameCounter++;
    }

    char c = (char) waitKey( 2 );
    if( c == 'q' )
      break;
    if( c == 'p' )
      paused = !paused;

  }

  return 0;
}
