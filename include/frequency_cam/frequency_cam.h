// -*-c++-*--------------------------------------------------------------------
// Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FREQUENCY_CAM__FREQUENCY_CAM_H_
#define FREQUENCY_CAM__FREQUENCY_CAM_H_

#include <event_camera_codecs/event_processor.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <eigen3/Eigen/Dense>

// #define DEBUG

namespace frequency_cam
{

class FrequencyCam : public event_camera_codecs::EventProcessor
{
public:
  FrequencyCam()
  : mean_position_csv_file_("mean_position_points.csv"),
    hough_circle_position_csv_file_("hough_circle_position_points.csv"),
    blob_detection_position_csv_file_("blob_detection_position_points.csv"),
    debug_image_counter_(0) {
    // Change thresholds
    blob_detector_params_.filterByColor = true;
    blob_detector_params_.blobColor = 255;

    // blob_detector_params_.minThreshold = 200;
    // blob_detector_params_.maxThreshold = 260;
     
    // Filter by Area.
    blob_detector_params_.filterByArea = true;
    blob_detector_params_.minArea = 50;
    blob_detector_params_.maxArea = 1000;
     
    // Filter by Circularity
    blob_detector_params_.filterByCircularity = true;
    blob_detector_params_.minCircularity = 0.3;
     
    // Filter by Convexity
    blob_detector_params_.filterByConvexity = true;
    blob_detector_params_.minConvexity = 0.4;
     
    // Filter by Inertia
    blob_detector_params_.filterByInertia = false;
    blob_detector_params_.minInertiaRatio = 0.5;

    // blob_detector_params_.minDistBetweenBlobs = 10;

    blob_detector_ = cv::SimpleBlobDetector::create(blob_detector_params_);
  }
  ~FrequencyCam();

  FrequencyCam(const FrequencyCam &) = delete;
  FrequencyCam & operator=(const FrequencyCam &) = delete;

  // ------------- inherited from EventProcessor
  inline void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override
  {
    // If the first time stamp is > 15s, there is an offset which we subtract every time.
    /*
    if (!initialize_time_stamps_) {
      initialize_time_stamps_ = true;
      if (sensor_time > 15000000000) {
        fix_time_stamps_ = true;
        std::cerr << "Time stamp needs to be fixed!" << std::endl;
        std::cerr << "First sensor time: " << sensor_time << std::endl;
        std::cerr << "First fixed sensor time: " << sensor_time - 16777215000 << std::endl;
      } else {
        std::cerr << "Time stamp seems fine!" << std::endl;
      }
    }
    if (fix_time_stamps_) {
      sensor_time -= 16777215000;
    }
    */
    // std::cout << "Event: time stamp: " << sensor_time << std::endl;
    // std::cerr << "ex: " << ex << ", ey: " << ey << ", time: " << shorten_time(sensor_time) << std::endl;
    Event e(shorten_time(sensor_time), ex, ey, polarity);
    updateState(&state_[e.y * width_ + e.x], e);
    lastEventTime_ = e.t;
    eventCount_++;
  }
  void eventExtTrigger(uint64_t /*sensor_time*/, uint8_t /*edge*/, uint8_t /*id*/) override
  {
    /*
    // If the first time stamp is > 15s, there is an offset which we subtract every time.
    if (!initialize_time_stamps_) {
      initialize_time_stamps_ = true;
      if (sensor_time > 15000000000) {
        fix_time_stamps_ = true;
      }
    }
    if (fix_time_stamps_) {
      sensor_time -= 16777215000;
    }
    // std::cout << "Trigger: time stamp: " << sensor_time << std::endl;
    if (!eventExtTriggerInitialized_) {
      lasteExternalEdge_ = edge;
      eventExtTriggerInitialized_ = true;
    } else {
      if (lasteExternalEdge_ == edge) {
        std::cerr << "Missed an external trigger edge" << std::endl;
      }
      // Take second event (falling edge) since this is the end of the exposure time
      // of the FB camera
      if (edge == 0) {
        sensor_time_ = sensor_time;
        hasValidTime_ = true;
        nrExtTriggers_++;
      }
      lasteExternalEdge_ = edge;
    }
    // std::cout << "External trigger: sensor_time: " << sensor_time << ", edge: " << std::to_string(edge) << std::endl;
    */
  }

  void finished() override {}
  void rawData(const char *, size_t) override {}
  // ------------- end of inherited from EventProcessor

  bool initialize(
    double minFreq, double maxFreq, double cutoffPeriod, int timeoutCycles, uint16_t debugX,
    uint16_t debugY, int visualization_choice);

  void initializeState(uint32_t width, uint32_t height, uint64_t t_first, uint64_t t_off);

  // returns frequency image
  cv::Mat makeFrequencyAndEventImage(
    cv::Mat * eventImage, bool overlayEvents, bool useLogFrequency, float dt, uint64_t trigger_timestamp = 0);

  void getStatistics(size_t * numEvents) const;
  void resetStatistics();

  void setTriggers(const std::string & triggers_file);

private:
  struct Event  // event representation for convenience
  {
    explicit Event(uint32_t ta = 0, uint16_t xa = 0, uint16_t ya = 0, bool p = false)
    : t(ta), x(xa), y(ya), polarity(p)
    {
    }
    // variables
    uint32_t t;
    uint16_t x;
    uint16_t y;
    bool polarity;
  };
  friend std::ostream & operator<<(std::ostream & os, const Event & e);

  struct Point
  {
    explicit Point(int xi = 0, int yi = 0)
    : x(xi), y(yi) {}
    int x;
    int y;
  };

  // define the per-pixel filter state
  typedef float variable_t;
  typedef uint32_t state_time_t;
  struct State
  {
    inline bool polarity() const { return (last_t_pol & (1 << 31)); }
    inline state_time_t lastTime() const { return (last_t_pol & ~(1 << 31)); }
    inline void set_time_and_polarity(state_time_t t, bool p)
    {
      last_t_pol = (static_cast<uint8_t>(p) << 31) | (t & ~(1 << 31));
    }
    // ------ variables
    state_time_t t_flip_up_down;  // time of last flip
    state_time_t t_flip_down_up;  // time of last flip
    variable_t L_km1;             // brightness lagged once
    variable_t L_km2;             // brightness lagged twice
    variable_t period;            // estimated period
    state_time_t last_t_pol;      // last polarity and time
  };

  inline void updateState(State * state, const Event & e)
  {
    State & s = *state;
    // raw change in polarity, will be 0 or +-1
    const float dp = e.polarity - s.polarity();
    // run the filter (see paper)
    const auto L_k = c_[0] * s.L_km1 + c_[1] * s.L_km2 + c_p_ * dp;
    if (L_k < 0 && s.L_km1 > 0) {
      // approximate reconstructed brightness crossed zero from above
      const float dt_ud = (e.t - s.t_flip_up_down) * 1e-6;  // "ud" = up_down
      if (dt_ud >= dtMin_ && dt_ud <= dtMax_) {
        // up-down (most accurate) dt is within valid range, use it!
        s.period = dt_ud;
      } else {
        // full period looks screwy, but maybe period can be computed from half cycle
        const float dt_du = (e.t - s.t_flip_down_up) * 1e-6;
        if (s.period > 0) {
          // If there already is a valid period established, check if it can be cleared out.
          // If it's not stale, don't update it because it would mean overriding a full-period
          // estimate with a half-period estimate.
          const float to = s.period * timeoutCycles_;  // timeout
          if (dt_ud > to && dt_du > 0.5 * to) {
            s.period = 0;  // stale pixel
          }
        } else {
          if (dt_du >= dtMinHalf_ && dt_du <= dtMaxHalf_) {
            // half-period estimate seems reasonable, make do with it
            s.period = 2 * dt_du;
          }
        }
      }
      s.t_flip_up_down = e.t;
    } else if (L_k > 0 && s.L_km1 < 0) {
      // approximate reconstructed brightness crossed zero from below
      const float dt_du = (e.t - s.t_flip_down_up) * 1e-6;  // "du" = down_up
      if (dt_du >= dtMin_ && dt_du <= dtMax_ && s.period <= 0) {
        // only use down-up transition if there is no established period
        // because it is less accurate than up-down transition
        s.period = dt_du;
      } else {
        const float dt_ud = (e.t - s.t_flip_up_down) * 1e-6;
        if (s.period > 0) {
          // If there already is a valid period established, check if it can be cleared out.
          // If it's not stale, don't update it because it would mean overriding a full-period
          // estimate with a half-period estimate.
          const float to = s.period * timeoutCycles_;  // timeout
          if (dt_du > to && dt_ud > 0.5 * to) {
            s.period = 0;  // stale pixel
          }
        } else {
          if (dt_ud >= dtMinHalf_ && dt_ud <= dtMaxHalf_) {
            // half-period estimate seems reasonable, make do with it
            s.period = 2 * dt_ud;
          }
        }
      }
      s.t_flip_down_up = e.t;
    }
#ifdef DEBUG
    if (e.x == debugX_ && e.y == debugY_) {
      const double dt = (e.t - std::max(s.t_flip_up_down, s.t_flip_down_up)) * 1e-6;
      debug_ << e.t + timeOffset_ << " " << dp << " " << L_k << " " << s.L_km1 << " " << s.L_km2
             << " " << dt << " " << s.period << " " << dtMin_ << " " << dtMax_ << std::endl;
    }
#endif
    s.L_km2 = s.L_km1;
    s.L_km1 = L_k;
    s.set_time_and_polarity(e.t, e.polarity);
  }

  struct NoTF
  {
    static double tf(double f) { return (f); }
    static double inv(double f) { return (f); }
  };
  struct LogTF
  {
    static double tf(double f) { return (std::log10(f)); }
    static double inv(double f) { return (std::pow(10.0, f)); }
  };
  struct EventFrameUpdater
  {
    static void update(cv::Mat * img, int ix, int iy, double dt, double dtMax)
    {
      if (dt < dtMax) {
        img->at<uint8_t>(iy, ix) = 255;
      }
    }
  };

  struct NoEventFrameUpdater
  {
    static void update(cv::Mat *, int, int, double, double) {}
  };

  template <class T, class U>
  cv::Mat makeTransformedFrequencyImage(cv::Mat * eventFrame, float eventImageDt, uint64_t trigger_timestamp)
  {
    nrRuns_++;

    std::vector<Point> frequency_points;

    // const int min_range_1 = 2800;
    // const int max_range_1 = 3200;
    // const int min_range_2 = 3800;
    // const int max_range_2 = 4200;
    const int min_range = 200;
    const int max_range = 800;
    cv::Mat rawImg(height_, width_, CV_32FC1, 0.0);
    const double maxDt = 1.0 / freq_[0] * timeoutCycles_;
    const double minFreq = T::tf(freq_[0]);
    for (uint32_t iy = 0; iy < height_; iy++) {
      for (uint32_t ix = 0; ix < width_; ix++) {
        const size_t offset = iy * width_ + ix;
        const State & state = state_[offset];
        // compute time since last touched
        const double dtEvent = (lastEventTime_ - state.lastTime()) * 1e-6;
        U::update(eventFrame, ix, iy, dtEvent, eventImageDt);
        if (x_updates_.at(ix) && y_updates_.at(iy)) {
        }
        if (state.period > 0) {
          const double dt =
            (lastEventTime_ - std::max(state.t_flip_up_down, state.t_flip_down_up)) * 1e-6;
          const double f = 1.0 / std::max(state.period, decltype(state.period)(1e-6));
          // filter out any pixels that have not been updated recently
          if (dt < maxDt * timeoutCycles_ && dt * f < timeoutCycles_) {
            auto frequency = std::max(T::tf(f), minFreq);
            // Only add points which are in our frequency range
            if (frequency > min_range && frequency < max_range) {
              frequency_points.emplace_back(ix, iy);
              // rawImg.at<float>(iy, ix) = frequency;
              rawImg.at<float>(iy, ix) = std::max(255, max_range);
            } else {
            // rawImg.at<float>(iy, ix) = frequency;
            rawImg.at<float>(iy, ix) = 0;  // mark as invalid
            }
          } else {
            rawImg.at<float>(iy, ix) = 0;  // mark as invalid
          }
        }
      }
    }

    // Save virtual frame for debbuging
    cv::Mat gray(height_, width_, CV_8UC1);
    rawImg.convertTo(gray, CV_8UC1);

    std::string string_counter = std::to_string(debug_image_counter_);
    unsigned int number_of_zeros = 5 - string_counter.length(); // add 2 zeros
    string_counter.insert(0, number_of_zeros, '0');

    std::string file_name = "debug_frames/debug_" + string_counter + ".png";
    cv::imwrite(file_name, gray);
    debug_image_counter_++;
    auto debug_position_color = cv::Scalar(100, 100, 100);

    // Hough circle detection
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1/*dp*/, 20/*minDist*/, 10/*param1*/, 8/*param2*/, 0, 10);

    if (3 == circles.size()) {
      std::vector<Point> circles_points;
      for (const auto& circle: circles) {
        circles_points.emplace_back(circle[0], circle[1]);
      }
      int idx_min;
      int idx_max;
      double dist_0_1;
      double dist_1_2;
      double dist_0_2;
      sort3Kp(circles_points, idx_min, idx_max, dist_0_1, dist_1_2, dist_0_2);

      hough_circle_position_csv_file_ << trigger_timestamp;
      for (const auto& circle : circles_points) {
        hough_circle_position_csv_file_ << ";" << cvRound(circle.x) << ";" << cvRound(circle.y);

        if (1 == visualization_choice_) {
          cv::Point center(cvRound(circle.x), cvRound(circle.y));
          // int radius = cvRound(circle[2]);
          // draw the circle center
          // cv::circle(rawImg, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0 );
          cv::circle(rawImg, center, 3, debug_position_color, 1, 8, 0);
          // draw the circle outline
          // cv::circle(rawImg, center, radius, cv::Scalar(800, 800, 800), 1, 8, 0);
        }
      }
      if (1 == visualization_choice_) {
        cv::putText(rawImg, "Nr. of markers: " + std::to_string(circles.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }
      hough_circle_position_csv_file_ << "\n";
      nrHoughDetectedWands_++;
    } else {
      // std::cout << "trigger_timestamp: " << trigger_timestamp << std::endl;
      hough_circle_position_csv_file_ << trigger_timestamp;
      hough_circle_position_csv_file_ << ";" << -1 << ";" << -1  << ";" << -1 << ";" << -1 << ";" << -1 << ";" << -1 << "\n";

      if (1 == visualization_choice_) {
        cv::putText(rawImg, "Nr. of markers: " + std::to_string(circles.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }
    }

    // Blob detection
    std::vector<cv::KeyPoint> keypoints;
    blob_detector_->detect(gray, keypoints);

    if (3 == keypoints.size()) {
      std::vector<Point> circles_points;
      for (const auto& keypoint: keypoints) {
        circles_points.emplace_back(keypoint.pt.x, keypoint.pt.y);
      }
      int idx_min;
      int idx_max;
      double dist_0_1;
      double dist_1_2;
      double dist_0_2;
      sort3Kp(circles_points, idx_min, idx_max, dist_0_1, dist_1_2, dist_0_2);

      blob_detection_position_csv_file_ << trigger_timestamp;
      // cv::drawKeypoints(rawImg, keypoints, rawImg, cv::Scalar(800, 800, 800), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      for (const auto& circle : circles_points) {
        blob_detection_position_csv_file_ << ";" << cvRound(circle.x) << ";" << cvRound(circle.y);
        if (2 == visualization_choice_) {
          cv::circle(rawImg, {cvRound(circle.x), cvRound(circle.y)}, 3, debug_position_color, 1);
        }
      }

      if (2 == visualization_choice_) {
        cv::putText(rawImg, "Nr. of markers: " + std::to_string(keypoints.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }
      blob_detection_position_csv_file_ << "\n";
      nrBlobDetectedWands_++;
    } else {
      // std::cout << "trigger_timestamp: " << trigger_timestamp << std::endl;
      blob_detection_position_csv_file_ << trigger_timestamp;
      blob_detection_position_csv_file_ << ";" << -1 << ";" << -1  << ";" << -1 << ";" << -1 << ";" << -1 << ";" << -1 << "\n";

      if (2 == visualization_choice_) {
        cv::putText(rawImg, "Nr. of markers: " + std::to_string(keypoints.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }
    }

    // Mean position
    std::vector<Point> filtered_frequency_points;
    std::vector<std::size_t> number_of_points;
    std::vector<std::size_t> assigned_indices;
    // Going through all the points which are in the frequency range
    // for the reference point
    for (std::size_t i = 0; i < frequency_points.size(); ++i) {
      // Skip if the point was already assigned to a cluster
      if (std::count(assigned_indices.begin(), assigned_indices.end(), i)) {
        continue;
      }

      std::vector<std::size_t> candidate_indices;
      std::vector<double> x_values;
      std::vector<double> y_values;
      auto x = frequency_points.at(i).x;
      auto y = frequency_points.at(i).y;

      std::size_t counts = 0;
      // Going through all the remaining points which are in the frequency range
      // for the candidate points
      for (std::size_t j = i + 1; j < frequency_points.size(); ++j) {
        // Skip if the point was already assigned to a cluster
        if (std::count(assigned_indices.begin(), assigned_indices.end(), j)) {
          continue;
        }

        auto x_candidate = frequency_points.at(j).x;
        auto y_candidate = frequency_points.at(j).y;

        // Make sure that points in the same cluster are close
        // double distance = 10;
        double distance = 8;
        if ((std::fabs(x - x_candidate) < distance) && (std::fabs(y - y_candidate) < distance)) {
          candidate_indices.emplace_back(j);
          x_values.emplace_back(x_candidate);
          y_values.emplace_back(y_candidate);
          counts++;
        }
      }

      // We want a certain amount of points for a cluster
      if (10 < counts) {
        candidate_indices.emplace_back(i);
        x_values.emplace_back(x);
        y_values.emplace_back(y);

        // Calculate mean position of the cluster
        auto mean_x = std::reduce(x_values.begin(), x_values.end());
        mean_x /= x_values.size();
        auto mean_y = std::reduce(y_values.begin(), y_values.end());
        mean_y /= y_values.size();

        // Centroid via Moments
        // double m00 = 0.0;
        // double m10 = 0.0;
        // double m01 = 0.0;
        // for (std::size_t x_i = 0; x_i < x_values.size(); ++x_i) {
        //   for (std::size_t y_i = 0; y_i < y_values.size(); ++y_i) {
        //      m00 += 1.0; 
        //      m10 += x_values.at(x_i); 
        //      m01 += y_values.at(y_i); 
        //   }
        // }
        // std::cout << "Mean x: " << mean_x << ", y: " << mean_y << std::endl;
        // mean_x = m10 / m00;
        // mean_y = m01 / m00;
        // std::cout << "Centroid x: " << mean_x << ", y: " << mean_y << std::endl;

        bool insert = true;
        for (const auto & point : filtered_frequency_points) {
          x = point.x;
          y = point.y;

          // Do not add cluster if its mean position is close to an already added
          // cluster
          // double cluster_distance = 20;
          double cluster_distance = 15;
          if ((std::fabs(x - mean_x) < cluster_distance) && (std::fabs(y - mean_y) < cluster_distance)) {
            insert = false;
            break;
          }
        }
        if (insert) {
          // filtered_frequency_points.emplace_back(mean_x, mean_y, frequency_point.first);
          filtered_frequency_points.emplace_back(mean_x, mean_y);
          number_of_points.emplace_back(x_values.size());
        }

        assigned_indices.insert(
          assigned_indices.end(), candidate_indices.begin(), candidate_indices.end());
      }
    }

    // Only proceed if we detected three clusters (three markers)
    // if (!filtered_frequency_points.empty()) {
    if (filtered_frequency_points.size() == 3) {
      int idx_min;
      int idx_max;
      double dist_0_1;
      double dist_1_2;
      double dist_0_2;
      sort3Kp(filtered_frequency_points, idx_min, idx_max, dist_0_1, dist_1_2, dist_0_2);

      // Check if points are on a line (are collinear)
      // double residual = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
      // double term_1 = std::get<0>(filtered_frequency_points.at(0)) * (std::get<1>(filtered_frequency_points.at(1)) - std::get<1>(filtered_frequency_points.at(2)));
      // double term_2 = std::get<0>(filtered_frequency_points.at(1)) * (std::get<1>(filtered_frequency_points.at(2)) - std::get<1>(filtered_frequency_points.at(0)));
      // double term_3 = std::get<0>(filtered_frequency_points.at(2)) * (std::get<1>(filtered_frequency_points.at(0)) - std::get<1>(filtered_frequency_points.at(1)));
      // double residual = std::get<0>(filtered_frequency_points.at(0)) * (std::get<1>(filtered_frequency_points.at(1)) - std::get<1>(filtered_frequency_points.at(2)))
      //                   + std::get<0>(filtered_frequency_points.at(1)) * (std::get<1>(filtered_frequency_points.at(2)) - std::get<1>(filtered_frequency_points.at(0)))
      //                   + std::get<0>(filtered_frequency_points.at(2)) * (std::get<1>(filtered_frequency_points.at(0)) - std::get<1>(filtered_frequency_points.at(1)));
      // std::cout << "x1: " << std::get<0>(filtered_frequency_points.at(0)) << std::endl;
      // std::cout << "y1 : " << std::get<1>(filtered_frequency_points.at(0)) << std::endl;
      // std::cout << "x2: " << std::get<0>(filtered_frequency_points.at(1)) << std::endl;
      // std::cout << "y2 : " << std::get<1>(filtered_frequency_points.at(1)) << std::endl;
      // std::cout << "x3: " << std::get<0>(filtered_frequency_points.at(2)) << std::endl;
      // std::cout << "y3 : " << std::get<1>(filtered_frequency_points.at(2)) << std::endl;
      // std::cout << "term 1: " << term_1 << ", term 2: " << term_2 << ", term_3: " << term_3 << std::endl;
      // std::cout << "residual: " << std::fabs(residual) << std::endl;


      // Line constraint from https://stackoverflow.com/questions/28619791/how-do-i-check-to-see-if-three-points-form-a-straight-line
      // Eigen::Matrix3f matrix;
      // matrix << std::get<0>(filtered_frequency_points.at(0)), std::get<1>(filtered_frequency_points.at(0)), 1.0,
      //           std::get<0>(filtered_frequency_points.at(1)), std::get<1>(filtered_frequency_points.at(1)), 1.0,
      //           std::get<0>(filtered_frequency_points.at(2)), std::get<1>(filtered_frequency_points.at(2)), 1.0;
      // Eigen::FullPivLU<Eigen::Matrix3f> lu_decomp(matrix);
      // auto rank = lu_decomp.rank();
      // std::cout << "Rank: " << rank << std::endl;
      // // result = rank([x2-x1, y2-y1; x3-x1, y3-y1]) < 2;

      // std::cout << "Filtered points:" << std::endl;
      // std::cout << "time stamp: " << lastEventTime_ << std::endl;
      //for (const auto & filtered_point : filtered_points) {

      // Write to csv file
      mean_position_csv_file_ << trigger_timestamp;
      for (std::size_t i = 0; i < filtered_frequency_points.size(); ++i) {
        // std::cout << "x: " << std::get<0>(filtered_frequency_points.at(i))
        //           << ", y: " << std::get<1>(filtered_frequency_points.at(i))
        //           << ", frequency: " << std::get<2>(filtered_frequency_points.at(i))
        //           << ", number of points: " << number_of_points.at(i) << std::endl;
        // auto frequency = std::get<0>(filtered_frequency_points.at(i));
        mean_position_csv_file_ << ";" << filtered_frequency_points.at(i).x << ";"
                  << filtered_frequency_points.at(i).y;

        // Visualize detected markers with circles
        if (3 == visualization_choice_) {
          double color_level = 0;
          if (i == 0) {
            color_level = 1000;
          } else if (i == 1) {
            color_level = 800;
          } else if (i == 2) {
            color_level = 600;
          }

          cv::circle(
            rawImg,
            {static_cast<int>(filtered_frequency_points.at(i).x),
             static_cast<int>(filtered_frequency_points.at(i).y)},
            // 2, CV_RGB(550, 550, 550), 4);
            12, CV_RGB(color_level, color_level, color_level), 6);
             //    cv::putText(rawImg, std::to_string(i),
             //                {static_cast<int>(std::get<0>(filtered_frequency_points.at(i))),
             //                 static_cast<int>(std::get<1>(filtered_frequency_points.at(i)))},
             //                cv::FONT_HERSHEY_SIMPLEX,
             //                2,
             //                550,
             //                4);
        }
      }
      mean_position_csv_file_ << "\n";

      if (3 == visualization_choice_) {
        cv::putText(rawImg, "Nr. of markers: " + std::to_string(filtered_frequency_points.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }

      // Add debug information to frame
      /*
      cv::putText(rawImg, "idx_min: " + std::to_string(idx_min), {50, 420}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "idx_max: " + std::to_string(idx_max), {50, 460}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "dist_0_1: " + std::to_string(dist_0_1), {50, 300}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "dist_1_2: " + std::to_string(dist_1_2), {50, 340}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "dist_0_2: " + std::to_string(dist_0_2), {50, 380}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);

      auto p1x = filtered_frequency_points.at(0).x;
      auto p1y = filtered_frequency_points.at(0).y;
      cv::putText(rawImg, "p0: x: " + std::to_string(p1x), {1000, 300}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "p0: y: " + std::to_string(p1y), {1000, 340}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      auto p2x = filtered_frequency_points.at(1).x;
      auto p2y = filtered_frequency_points.at(1).y;
      cv::putText(rawImg, "p1: x: " + std::to_string(p2x), {1000, 380}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "p1: y: " + std::to_string(p2y), {1000, 420}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      auto p3x = filtered_frequency_points.at(2).x;
      auto p3y = filtered_frequency_points.at(2).y;
      cv::putText(rawImg, "p2: x: " + std::to_string(p3x), {1000, 460}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      cv::putText(rawImg, "p2: y: " + std::to_string(p3y), {1000, 500}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      */

      nrMeanDetectedWands_++;
    } else {
      // std::cout << "trigger_timestamp: " << trigger_timestamp << std::endl;
      mean_position_csv_file_ << trigger_timestamp;
      mean_position_csv_file_ << ";" << -1 << ";" << -1  << ";" << -1 << ";" << -1 << ";" << -1 << ";" << -1 << "\n";

      if (3 == visualization_choice_) {
        cv::putText(rawImg, "Nr. of markers: " + std::to_string(filtered_frequency_points.size()), {100, 100}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
      }
    }

    cv::putText(rawImg, "time stamp: " + std::to_string(trigger_timestamp), {800, 600}, cv::FONT_HERSHEY_SIMPLEX, 1, 550, 4);
    return (rawImg);
  }

  static inline uint32_t shorten_time(uint64_t t)
  {
    return (static_cast<uint32_t>((t / 1000) & 0xFFFFFFFF));
  }

  void sort3Kp(std::vector<Point>& kp, int& idx_min, int& idx_max, double& dist_0_1, double& dist_1_2, double& dist_0_2);

  // ------ variables ----
  State * state_{0};
  double freq_[2]{-1.0, -1.0};  // frequency range
  double tfFreq_[2]{0, 1.0};    // transformed frequency range
  uint32_t width_{0};           // image width
  uint32_t height_{0};          // image height
  uint64_t eventCount_{0};
  uint32_t lastEventTime_;
  // ---------- variables for state update
  variable_t c_[2];
  variable_t c_p_{0};
  variable_t dtMin_{0};
  variable_t dtMax_{1.0};
  variable_t dtMinHalf_{0};
  variable_t dtMaxHalf_{0.5};
  variable_t timeoutCycles_{2.0};  // how many silent cycles until freq is invalid
  //
  // ------------------ debugging stuff
  std::ofstream debug_;
  uint16_t debugX_{0};
  uint16_t debugY_{0};
  std::atomic<bool> hasValidTime_{false};
  uint64_t timeOffset_{0};
  uint64_t sensor_time_;
  bool eventExtTriggerInitialized_{false};
  std::size_t nrExtTriggers_{0};
  std::size_t nrSyncMatches_{0};
  std::size_t nrMeanDetectedWands_{0};
  std::size_t nrHoughDetectedWands_{0};
  std::size_t nrBlobDetectedWands_{0};
  std::size_t nrRuns_{0};
  uint8_t lasteExternalEdge_;

  std::ofstream mean_position_csv_file_;
  std::ofstream hough_circle_position_csv_file_;
  std::ofstream blob_detection_position_csv_file_;
  std::vector<uint64_t> externalTriggers_;
  bool initialize_time_stamps_{false};
  bool fix_time_stamps_{false}; 

  cv::Ptr<cv::SimpleBlobDetector> blob_detector_;
  cv::SimpleBlobDetector::Params blob_detector_params_;

  std::size_t debug_image_counter_;
  std::vector<int> x_updates_;
  std::vector<int> y_updates_;

  std::size_t visualization_choice_;
};
std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e);
}  // namespace frequency_cam
#endif  // FREQUENCY_CAM__FREQUENCY_CAM_H_
