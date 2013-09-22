Tracker Algorithms
==================

.. highlight:: cpp

The following algorithms are implemented at the moment.

.. [MIL] B Babenko, M-H Yang, and S Belongie, Visual Tracking with Online Multiple Instance Learning, In CVPR, 2009

.. [OLB] H Grabner, M Grabner, and H Bischof, Real-time tracking via on-line boosting, In Proc. BMVC, volume 1, pages 47– 56, 2006

TrackerBoosting
---------------

This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.
The classifier uses the surrounding background as negative examples in update step to avoid the drifting problem. The implementation is based on
[OLB]_.

.. ocv:class:: TrackerBoosting

Implementation of TrackerBoosting from :ocv:class:`Tracker`::

   class CV_EXPORTS_W TrackerBoosting : public Tracker
   {
    public:

     TrackerBoosting( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() );

     virtual ~TrackerBoosting();

     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;


   };

TrackerBoosting::Params
-----------------------------------------------------------------------

.. ocv:struct:: TrackerBoosting::Params

List of BOOSTING parameters::

   struct CV_EXPORTS Params
   {
    Params();
    int numClassifiers;  //the number of classifiers to use in a OnlineBoosting algorithm
    float samplerOverlap;  //search region parameters to use in a OnlineBoosting algorithm
    float samplerSearchFactor;  // search region parameters to use in a OnlineBoosting algorithm
    int iterationInit;  //the initial iterations
    int featureSetNumFeatures;  // #features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
   };

TrackerBoosting::TrackerBoosting
-----------------------------------------------------------------------

Constructor

.. ocv:function:: bool TrackerBoosting::TrackerBoosting( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() )

    :param parameters: BOOSTING parameters :ocv:struct:`TrackerBoosting::Params`

TrackerMIL
----------

The MIL algorithm trains a classifier in an online manner to separate the object from the background. Multiple Instance Learning avoids the drift problem for a robust tracking. The implementation is based on [MIL]_.

Original code can be found here http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

.. ocv:class:: TrackerMIL

Implementation of TrackerMIL from :ocv:class:`Tracker`::

   class CV_EXPORTS_W TrackerMIL : public Tracker
   {
    public:

     TrackerMIL( const TrackerMIL::Params &parameters = TrackerMIL::Params() );

     virtual ~TrackerMIL();

     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;

   };

TrackerMIL::Params
------------------

.. ocv:struct:: TrackerMIL::Params

List of MIL parameters::

   struct CV_EXPORTS Params
   {
    Params();
    //parameters for sampler
    float samplerInitInRadius;   // radius for gathering positive instances during init
    int samplerInitMaxNegNum;    // # negative samples to use during init
    float samplerSearchWinSize;  // size of search window
    float samplerTrackInRadius;  // radius for gathering positive instances during tracking
    int samplerTrackMaxPosNum;   // # positive samples to use during tracking
    int samplerTrackMaxNegNum;   // # negative samples to use during tracking

    int featureSetNumFeatures;   // # features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
   };

TrackerMIL::TrackerMIL
----------------------

Constructor

.. ocv:function:: bool TrackerMIL::TrackerMIL( const TrackerMIL::Params &parameters = TrackerMIL::Params() )

    :param parameters: MIL parameters :ocv:struct:`TrackerMIL::Params`

TrackerPF
----------

This tracker is based on a particle filtering algorithm.

.. ocv:class:: TrackerPF

Implementation of TrackerPF from :ocv:class:`Tracker`::

   class CV_EXPORTS_W TrackerPF : public Tracker
   {
    public:

     TrackerPF( const TrackerPF::Params &parameters = TrackerPF::Params() );

     virtual ~TrackerPF();

     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;

   };

TrackerPF::Params
------------------

.. ocv:struct:: TrackerPF::Params

List of PF parameters::

   struct CV_EXPORTS Params
   {
     Params();
     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;

     int iterationNum;
     int particlesNum;
     double alpha;
     Mat_<double> std; 
   };
