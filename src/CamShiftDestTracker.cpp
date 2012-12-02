//Includes all the headers necessary to use the most common public pieces of the ROS system.
#include <ros/ros.h>
//Use image_transport for publishing and subscribing to images in ROS
#include <image_transport/image_transport.h>
//Use cv_bridge to convert between ROS and OpenCV Image formats
#include <cv_bridge/cv_bridge.h>
//Include some useful constants for image encoding. Refer to: http://www.ros.org/doc/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html for more info.
#include <sensor_msgs/image_encodings.h>
//Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
//Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>

using namespace cv;
using namespace std;

//Store all constants for image encodings in the enc namespace to be used later.
namespace enc = sensor_msgs::image_encodings;


//Use method of ImageTransport to create image publisher
//image_transport::Publisher pub;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
static Point origin;
static Rect selection;
int vmin = 10, vmax = 256, smin = 45;
Mat image;
static int imgWidth;


static ros::Publisher robotAngleVar;
static ros::Publisher robotRchdDest;

void camShift(Mat inImg);


static void onMouse( int event, int x, int y, int, void* )
{
    //ROS_INFO("Mouse detected");
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}

//This function is called everytime a new image_info message is published
void camInfoCallback(const sensor_msgs::CameraInfo & camInfoMsg)
{
  //Store the image width for calculation of angle
  imgWidth = camInfoMsg.width;
}

void destCoordCallback(const sensor_msgs::RegionOfInterest& destROI)
{
  //Copy the ROI to the local buffer
  selection.x = destROI.x_offset;
  selection.y = destROI.y_offset;
  selection.height = destROI.height;
  selection.width = destROI.width;
  trackObject = -1;
}


//This function is called everytime a new image is published
void imageCallback(const sensor_msgs::ImageConstPtr& original_image)
{
	//Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		//Always copy, returning a mutable CvImage
		//OpenCV expects color images to use BGR channel order.
		cv_ptr = cv_bridge::toCvCopy(original_image, enc::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		//if there is an error during conversion, display it
		ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
		return;
	}
	camShift(cv_ptr->image);
}


void trackArea(Rect window)
{
  //TBD
  //Code should track the area of the target. If the area grows very fast then most probably the destination has a shift
  //And it has lost the destination
}

void calcAngle(Point2f destCentre)
{
  std_msgs::Float32 normAngle;
  //If we have started tracking the object
  if(!trackObject)
  {
    normAngle.data = (destCentre.x - ((float)imgWidth/2))/((float)imgWidth/2);
    robotAngleVar.publish(normAngle);
  }
}

void camShift(Mat inImg)
{
  static Rect trackWindow;
  static int hsize = 16;
  static float hranges[] = {0,180};
  static const float* phranges = hranges;
  static Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
  static bool paused = false;
  RotatedRect trackBox;

  //If the image processing is not paused
  if( !paused )
  {
    //cap >> frame;
    if( inImg.empty() )
    {
      ROS_INFO("Camera image empty");
      return;//break;
    }
  }

  //Use the input image as the reference
  //Only a shallow copy, so relatively fast
  image = inImg;

  if(!paused)
  {
      //Convert the colour space to HSV
      cvtColor(image, hsv, CV_BGR2HSV);

      //If the destination coordinates have been received, then start the tracking
      //trackObject is set when the destination coordinates have been received
      if( trackObject )
      {
          int _vmin = vmin, _vmax = vmax;

          inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                  Scalar(180, 256, MAX(_vmin, _vmax)), mask);
          int ch[] = {0, 0};
          hue.create(hsv.size(), hsv.depth());
          mixChannels(&hsv, 1, &hue, 1, ch, 1);

          //Do the following steps only for the first time
          if( trackObject < 0 )
          {
              //Set the Region of interest and the mask for it
              Mat roi(hue, selection), maskroi(mask, selection);
              //Calculate the histogram of this
              calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
              normalize(hist, hist, 0, 255, CV_MINMAX);

              trackWindow = selection;
              trackObject = 1;

              histimg = Scalar::all(0);
              int binW = histimg.cols / hsize;
              Mat buf(1, hsize, CV_8UC3);
              for( int i = 0; i < hsize; i++ )
                  buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
              cvtColor(buf, buf, CV_HSV2BGR);

              for( int i = 0; i < hsize; i++ )
              {
                  int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                  rectangle( histimg, Point(i*binW,histimg.rows),
                             Point((i+1)*binW,histimg.rows - val),
                             Scalar(buf.at<Vec3b>(i)), -1, 8 );
              }
          }

          calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
          backproj &= mask;
          trackBox = CamShift(backproj, trackWindow,
                              TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
          if( trackWindow.area() <= 1 )
          {
              ROS_INFO("*********DESTINATION LOST************");
              ROS_INFO("track height %d width %d", trackWindow.height, trackWindow.width);
              trackObject = 0; //Disable tracking to avoid termination of node due to negative heights TBD
              int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
              trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                 trackWindow.x + r, trackWindow.y + r) &
                            Rect(0, 0, cols, rows);
          }

          if( backprojMode )
              cvtColor( backproj, image, CV_GRAY2BGR );
          ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
      }
  }
  else if( trackObject < 0 )
  {
    //If a new destination has been selected stop pausing
    paused = false;
  }

  //Code to display an inverted image of the selected region
  //Remove this in the fall validation expt TBD
  if( selectObject && selection.width > 0 && selection.height > 0 )
  {
      Mat roi(image, selection);
      bitwise_not(roi, roi);
  }

  imshow( "CamShift Demo", image );
  imshow( "Histogram", histimg );

  char c = (char)waitKey(1);
  if( c == 27 )
      ROS_INFO("Exit boss");//break;
  switch(c)
  {
  case 'b':
      backprojMode = !backprojMode;
      break;
  case 'c':
      trackObject = 0;
      histimg = Scalar::all(0);
      break;
  case 'h':
      showHist = !showHist;
      if( !showHist )
          destroyWindow( "Histogram" );
      else
          namedWindow( "Histogram", 1 );
      break;
  case 'p':
      paused = !paused;
      break;
  default:
      break;
  }
  setMouseCallback( "CamShift Demo", onMouse, 0 );
  createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
  createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
  createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );


  trackArea(trackWindow);
  //Find the angle of the destination wrt to the robot and publish that
  calcAngle(trackBox.center);
}




/**
* This is ROS node to track the destination image
*/
int main(int argc, char **argv)
{
	/**
	* The ros::init() function needs to see argc and argv so that it can perform
	* any ROS arguments and name remapping that were provided at the command line. For programmatic
	* remappings you can use a different version of init() which takes remappings
	* directly, but for most command-line programs, passing argc and argv is the easiest
	* way to do it.  The third argument to init() is the name of the node. Node names must be unique in a running system.
	* The name used here must be a base name, ie. it cannot have a / in it.
	* You must call one of the versions of ros::init() before using any other
	* part of the ROS system.
	*/
    ros::init(argc, argv, "image_processor");
    ROS_INFO("-----------------");
	/**
	* NodeHandle is the main access point to communications with the ROS system.
	* The first NodeHandle constructed will fully initialize this node, and the last
	* NodeHandle destructed will close down the node.
	*/
    ros::NodeHandle nh;
	//Create an ImageTransport instance, initializing it with our NodeHandle.
    image_transport::ImageTransport it(nh);



	//OpenCV HighGUI call to create a display window on start-up.
	namedWindow( "Histogram", 0 );
	namedWindow( "CamShift Demo", 0 );

	/**
	* Subscribe to the "camera/image_raw" base topic. The actual ROS topic subscribed to depends on which transport is used.
	* In the default case, "raw" transport, the topic is in fact "camera/image_raw" with type sensor_msgs/Image. ROS will call
	* the "imageCallback" function whenever a new image arrives. The 2nd argument is the queue size.
	* subscribe() returns an image_transport::Subscriber object, that you must hold on to until you want to unsubscribe.
	* When the Subscriber object is destructed, it will automaticaInfoCallbacklly unsubscribe from the "camera/image_raw" base topic.
	*/
    image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, imageCallback);
    ros::Subscriber camInfo = nh.subscribe("camera/camera_info", 1, camInfoCallback);
    ros::Subscriber destCoord = nh.subscribe("dest_coord", 1, destCoordCallback);
    robotAngleVar = nh.advertise<std_msgs::Float32>("robot_angle_variation", 100);
    robotRchdDest = nh.advertise<std_msgs::Bool>("robot_reached_destination", 10);


	//OpenCV HighGUI call to destroy a display window on shut-down.
	//destroyWindow(WINDOW);
    destroyWindow("Histogram");
    destroyWindow("CamShift Demo");


	/**
	* In this application all user callbacks will be called from within the ros::spin() call.
	* ros::spin() will not return until the node has been shutdown, either through a call
	* to ros::shutdown() or a Ctrl-C.
	*/
    ros::spin();

	//ROS_INFO is the replacement for printf/cout.
	ROS_INFO("tutorialROSOpenCV::main.cpp::No error.");
}
