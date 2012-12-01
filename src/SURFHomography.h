/*
 * SURF_Homography.h
 *
 *  Created on: Nov 30, 2012
 *      Author: ashok
 */

#ifndef SURF_HOMOGRAPHY_H_
#define SURF_HOMOGRAPHY_H_

#include "stdio.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

extern int SURF_main(Mat img_scene, Mat img_object);


#endif /* SURF_HOMOGRAPHY_H_ */
