/***************************************************************************

    file                 : rex.cpp
    created              : Nov 2016
    copyright            : (C) 2016 Rex Ying

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef _WIN32
#include <windows.h>
#endif

#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <stdio.h>
#include <stdlib.h> 
#include <string>
#include <sstream>
#include <string.h> 
#include <math.h>
#include <ctime>
#include <chrono>

#include <tgf.h> 
#include <track.h> 
#include <car.h> 
#include <raceman.h> 
#include <robottools.h>
#include <robot.h>

static tTrack	*curTrack;

static void initTrack(int index, tTrack* track, void *carHandle, void **carParmHandle, tSituation *s); 
static void newrace(int index, tCarElt* car, tSituation *s); 
static void drive(int index, tCarElt* car, tSituation *s); 
static void endrace(int index, tCarElt *car, tSituation *s);
static void shutdown(int index);
static int  InitFuncPt(int index, void *pt); 

extern bool toggle_export;
extern std::string img_export_path;
//extern std::string sensor_export_path;
std::string sensor_export_path = "/home/rex/workspace/torcs-data/sensor/";
static const int NUM_SENSOR_VALS = 4;
static std::ofstream sensor_output_file;

static const int COUNT_BEFORE_SAVE = 1;
static int drive_count = 1;

static long prev_time = 0;

/* 
 * Module entry point  */ 
extern "C" int rexy(tModInfo *modInfo) 
{
    memset(modInfo, 0, 10*sizeof(tModInfo));

    modInfo->name    = strdup("rexy 0");		/* name of the module (short) */
    modInfo->desc    = strdup("");	/* description of the module (can be long) */
    modInfo->fctInit = InitFuncPt;		/* init function */
    modInfo->gfId    = ROB_IDENT;		/* supported framework version */
    modInfo->index   = 1;

    return 0; 
} 

/* Module interface initialization. */
static int 
InitFuncPt(int index, void *pt) 
{ 
    tRobotItf *itf  = (tRobotItf *)pt; 

    itf->rbNewTrack = initTrack; /* Give the robot the track view called */ 
				 /* for every track change or new race */ 
    itf->rbNewRace  = newrace; 	 /* Start a new race */
    itf->rbDrive    = drive;	 /* Drive during race */
    itf->rbPitCmd   = NULL;
    itf->rbEndRace  = endrace;	 /* End of the current race */
    itf->rbShutdown = shutdown;	 /* Called before the module is unloaded */
    itf->index      = index; 	 /* Index used if multiple interfaces */
    return 0; 
} 

/* Called for every track change or new race. */
static void  
initTrack(int index, tTrack* track, void *carHandle, void **carParmHandle, tSituation *s) 
{ 
    curTrack = track;
    *carParmHandle = NULL; 
} 


long getTimeString()
{
  //std::time_t curr_time = std::time(nullptr);
  //ss << curr_time;
  // 0.1s precision
  long curr_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  return curr_time;
}



/* Start a new race. */
static void  
newrace(int index, tCarElt* car, tSituation *s) 
{ 
  std::stringstream ss;
  ss << sensor_export_path << getTimeString() << ".txt";
  std::string fname = ss.str();
  sensor_output_file.open(fname.c_str(), std::fstream::trunc | std::fstream::out);
  sensor_output_file << "";
  sensor_output_file.close();
  sensor_output_file.open(fname.c_str(), std::fstream::app);
} 


/* Compute gear. */
const float SHIFT = 0.85;         /* [-] (% of rpmredline) */
const float SHIFT_MARGIN = 4.0;  /* [m/s] */

int getGear(tCarElt *car)
{
	if (car->_gear <= 0)
		return 1;
	float gr_up = car->_gearRatio[car->_gear + car->_gearOffset];
	float omega = car->_enginerpmRedLine/gr_up;
	float wr = car->_wheelRadius(2);

	if (omega*wr*SHIFT < car->_speed_x) {
		return car->_gear + 1;
	} else {
		float gr_down = car->_gearRatio[car->_gear + car->_gearOffset - 1];
		omega = car->_enginerpmRedLine/gr_down;
		if (car->_gear > 1 && omega*wr*SHIFT > car->_speed_x + SHIFT_MARGIN) {
			return car->_gear - 1;
		}
	}
	return car->_gear;
}


/* check if the car is stuck */
const float MAX_UNSTUCK_SPEED = 5.0;   /* [m/s] */
const float MIN_UNSTUCK_DIST = 3.0;    /* [m] */
const float MAX_UNSTUCK_ANGLE = 20.0/180.0*PI;
const int MAX_UNSTUCK_COUNT = 250;
static int stuck = 0;

bool isStuck(tCarElt* car)
{
    float angle = RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw;
    NORM_PI_PI(angle);

    if (fabs(angle) > MAX_UNSTUCK_ANGLE &&
        car->_speed_x < MAX_UNSTUCK_SPEED &&
        fabs(car->_trkPos.toMiddle) > MIN_UNSTUCK_DIST) {
        if (stuck > MAX_UNSTUCK_COUNT && car->_trkPos.toMiddle*angle < 0.0) {
            return true;
        } else {
            stuck++;
            return false;
        }
    } else {
        stuck = 0;
        return false;
    }
}


/* Drive during race. */

double max_speed=60 / 3.6;
double min_speed = 15 / 3.6;
//double keepLR=-2.0;   // for two-lane
double keepLR=0.0;   // for three-lane

static double lane_offsets[] = {-4.0, 0, 4.0};
static int curr_offset_idx = 2;

static int COUNT_BEFORE_CHANGE_ACC = 300;
static int curr_count_change_acc = 1;
static double acc = 0.0;

static void save(double* vals, int num_vals) {

  long curr_time = getTimeString();
  if (prev_time != 0 && (curr_time - prev_time < 20))
  {
    return;
  }
  prev_time = curr_time;

  sensor_output_file << curr_time << " ";
  sensor_output_file << std::fixed;
  for (int i = 0; i < num_vals; i++) {
    sensor_output_file << vals[i] << std::setprecision(4) << " "; 
  }
  sensor_output_file << std::endl;
}


static void drive(int index, tCarElt* car, tSituation *s) 
{ 
    memset(&car->ctrl, 0, sizeof(tCarCtrl));

    if (curr_count_change_acc == COUNT_BEFORE_CHANGE_ACC)
    {
      curr_count_change_acc = 1;
      
      std::default_random_engine generator(std::random_device{}());
      std::normal_distribution<double> distribution(0.1, 0.3);
      acc = distribution(generator);

      double acc_threshold = 0.7;
      double brake_threshold = 0.5;
      if (acc > acc_threshold) 
      {
        acc = acc_threshold;
      }
      if (acc < -brake_threshold)
      {
        acc = -brake_threshold;
      }
    }
    else {
      curr_count_change_acc++;
    }

    if (isStuck(car)) {
        float angle = -RtTrackSideTgAngleL(&(car->_trkPos)) + car->_yaw;
        NORM_PI_PI(angle); // put the angle back in the range from -PI to PI

        car->ctrl.steer = angle / car->_steerLock;
        car->ctrl.gear = -1; // reverse gear
        car->ctrl.accelCmd = 0.3; // 30% accelerator pedal
        car->ctrl.brakeCmd = 0.0; // no brakes
    } 
    else {
        float angle;
        const float SC = 1.0;

        angle = RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw;
        NORM_PI_PI(angle); // put the angle back in the range from -PI to PI
        angle -= SC*(car->_trkPos.toMiddle + lane_offsets[curr_offset_idx])/car->_trkPos.seg->width;

        // set up the values to return
        car->ctrl.steer = angle / car->_steerLock;
        car->ctrl.gear = getGear(car);

        if (car->_speed_x>max_speed) {
           car->ctrl.brakeCmd=0.5;
           car->ctrl.accelCmd=0.0;
        }
        else if  (car->_speed_x<min_speed) {
           car->ctrl.accelCmd=0.5;
           car->ctrl.brakeCmd=0.0;
        }
        else if (acc > 0)
        {
          car->ctrl.accelCmd = acc;
          car->ctrl.brakeCmd = 0.0;
        }
        else
        {
          car->ctrl.accelCmd = 0.0;
          car->ctrl.brakeCmd = acc;
        }
        printf("acc %f\n", car->ctrl.accelCmd);
    }    

    
    if (toggle_export)
    {
      if (drive_count == COUNT_BEFORE_SAVE) {
        /*
        printf("H speedX = %f\n", car->_speed_x);
        printf("H speedY = %f\n", car->_speed_y);
        printf("H accelX = %f\n", car->_accel_x);
        printf("H accelY = %f\n", car->_accel_y);
        */

        double d[NUM_SENSOR_VALS] = {car->_speed_x, car->_speed_y, car->_accel_x, car->_accel_y};
        save(d, NUM_SENSOR_VALS);
        drive_count = 1;
      }
      else {
        drive_count++;
      }
    }
}

/* End of the current race */
static void
endrace(int index, tCarElt *car, tSituation *s)
{
  sensor_output_file.close();
  std::cout << "Race ends" << std::endl;
}

/* Called before the module is unloaded */
static void
shutdown(int index)
{
}

