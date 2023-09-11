// -*-c++-*----------------------------------------------------------------------------------------
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

#include "frequency_cam/frequency_cam.h"

#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>

template <typename T, typename A> int arg_max(std::vector<T, A> const &vec) {
  return static_cast<int>(
      std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T, typename A> int arg_min(std::vector<T, A> const &vec) {
  return static_cast<int>(
      std::distance(vec.begin(), min_element(vec.begin(), vec.end())));
}

namespace frequency_cam
{

FrequencyCam::~FrequencyCam() {
  mean_position_csv_file_.close();
  hough_circle_position_csv_file_.close();
  blob_detection_position_csv_file_.close();
  delete[] state_;

  std::cout << "Number of external triggers: " << nrExtTriggers_ << std::endl;
  std::cout << "Number of time synchronization matches: " << nrSyncMatches_ << std::endl;
  std::cout << "Number of detected wands by the mean approach: " << nrMeanDetectedWands_ << std::endl;
  std::cout << "Number of detected wands by the hough approach: " << nrHoughDetectedWands_ << std::endl;
  std::cout << "Number of detected wands by the blog approach: " << nrBlobDetectedWands_ << std::endl;
}

static void compute_alpha_beta(const double T_cut, double * alpha, double * beta)
{
  // compute the filter coefficients alpha and beta (see paper)
  const double omega_cut = 2 * M_PI / T_cut;
  const double phi = 2 - std::cos(omega_cut);
  *alpha = (1.0 - std::sin(omega_cut)) / std::cos(omega_cut);
  *beta = phi - std::sqrt(phi * phi - 1.0);  // see paper
}

bool FrequencyCam::initialize(
  double minFreq, double maxFreq, double cutoffPeriod, int timeoutCycles, uint16_t debugX,
  uint16_t debugY, int visualization_choice)
{
#ifdef DEBUG  // the debug flag must be set in the header file
  debug_.open("freq.txt", std::ofstream::out);
#endif

  freq_[0] = std::max(minFreq, 0.1);
  freq_[1] = maxFreq;
  dtMax_ = 1.0 / freq_[0];
  dtMaxHalf_ = 0.5 * dtMax_;
  dtMin_ = 1.0 / (freq_[1] >= freq_[0] ? freq_[1] : 1.0);
  dtMinHalf_ = 0.5 * dtMin_;
  timeoutCycles_ = timeoutCycles;
  const double T_prefilter = cutoffPeriod;
  double alpha_prefilter, beta_prefilter;
  compute_alpha_beta(T_prefilter, &alpha_prefilter, &beta_prefilter);

  // compute IIR filter coefficient from alpha and beta (see paper)
  c_[0] = alpha_prefilter + beta_prefilter;
  c_[1] = -alpha_prefilter * beta_prefilter;
  c_p_ = 0.5 * (1 + beta_prefilter);

  debugX_ = debugX;
  debugY_ = debugY;

  visualization_choice_ = visualization_choice;

  return (true);
}

void FrequencyCam::initializeState(uint32_t width, uint32_t height, uint64_t t_full, uint64_t t_off)
{
  const uint32_t t = shorten_time(t_full) - 1;
#ifdef DEBUG
  timeOffset_ = (t_off / 1000) - shorten_time(t_off);  // safe high bits lost by shortening
#else
  (void)t_off;
#endif

  for (std::size_t i = 0; i < width; ++i) {
    x_updates_.emplace_back(false);
  }
  std::cerr << "x_updates_.size(): " << x_updates_.size() << std::endl;
  for (std::size_t i = 0; i < height; ++i) {
    y_updates_.emplace_back(false);
  }
  std::cerr << "y_updates_.size(): " << y_updates_.size() << std::endl;

  width_ = width;
  height_ = height;
  state_ = new State[width * height];
  for (size_t i = 0; i < width * height; i++) {
    State & s = state_[i];
    s.t_flip_up_down = t;
    s.t_flip_down_up = t;
    s.L_km1 = 0;
    s.L_km2 = 0;
    s.period = -1;
    s.set_time_and_polarity(t, 0);
  }
}

cv::Mat FrequencyCam::makeFrequencyAndEventImage(
  cv::Mat * evImg, bool overlayEvents, bool useLogFrequency, float dt, uint64_t trigger_timestamp)
{
  if (overlayEvents) {
    *evImg = cv::Mat::zeros(height_, width_, CV_8UC1);
  }
  if (useLogFrequency) {
    return (
      overlayEvents ? makeTransformedFrequencyImage<LogTF, EventFrameUpdater>(evImg, dt, trigger_timestamp)
                    : makeTransformedFrequencyImage<LogTF, NoEventFrameUpdater>(evImg, dt, trigger_timestamp));
  }
  return (
    overlayEvents ? makeTransformedFrequencyImage<NoTF, EventFrameUpdater>(evImg, dt, trigger_timestamp)
                  : makeTransformedFrequencyImage<NoTF, NoEventFrameUpdater>(evImg, dt, trigger_timestamp));
}

void FrequencyCam::getStatistics(size_t * numEvents) const { *numEvents = eventCount_; }

void FrequencyCam::resetStatistics() { eventCount_ = 0; }

// void FrequencyCam::sort3Kp(vector<cv::KeyPoint> &kp) {
void FrequencyCam::sort3Kp(std::vector<Point>& kp, int& idx_min, int& idx_max, double& dist_0_1, double& dist_1_2, double& dist_0_2) {
  // Sorts from closest together to most seperated
  // vector<cv::KeyPoint> cp_kp;
  std::vector<Point> cp_kp;
  cp_kp = kp;
  std::vector<double> d;
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = i + 1; j < 3; j++) {
      // cv::Point2f diff = kp.at(i).pt - kp.at(j).pt;
      cv::Point2d diff {static_cast<double>(kp.at(i).x - kp.at(j).x),
                        static_cast<double>(kp.at(i).y - kp.at(j).y)};
      double dist = sqrt(diff.x * diff.x + diff.y * diff.y);
      d.push_back(dist);
    }
  }

  // int idx_min = arg_min(d);
  // int idx_max = arg_max(d);
  idx_min = arg_min(d);
  idx_max = arg_max(d);
  switch (idx_max) {
  case 0:
    kp.at(1) = cp_kp.at(2);
    if (idx_min == 1) {
      kp.at(0) = cp_kp.at(0);
      kp.at(2) = cp_kp.at(1);
    } else {
      kp.at(0) = cp_kp.at(1);
      kp.at(2) = cp_kp.at(0);
    }
    break;
  case 1:
    kp.at(1) = cp_kp.at(1);
    if (idx_min == 0) {
      kp.at(0) = cp_kp.at(0);
      kp.at(2) = cp_kp.at(2);
    } else {
      kp.at(0) = cp_kp.at(2);
      kp.at(2) = cp_kp.at(0);
    }
    break;
  case 2:
    kp.at(1) = cp_kp.at(0);
    if (idx_min == 0) {
      kp.at(0) = cp_kp.at(1);
      kp.at(2) = cp_kp.at(2);
    } else {
      kp.at(0) = cp_kp.at(2);
      kp.at(2) = cp_kp.at(1);
    }
    break;
  }

  cv::Point2d diff_0_1 {static_cast<double>(kp.at(0).x - kp.at(1).x),
                        static_cast<double>(kp.at(0).y - kp.at(1).y)};
  dist_0_1 = sqrt(diff_0_1.x * diff_0_1.x + diff_0_1.y * diff_0_1.y);
  cv::Point2d diff_1_2 {static_cast<double>(kp.at(1).x - kp.at(2).x),
                        static_cast<double>(kp.at(1).y - kp.at(2).y)};
  dist_1_2 = sqrt(diff_1_2.x * diff_1_2.x + diff_1_2.y * diff_1_2.y);
  cv::Point2d diff_0_2 {static_cast<double>(kp.at(0).x - kp.at(2).x),
                        static_cast<double>(kp.at(0).y - kp.at(2).y)};
  dist_0_2 = sqrt(diff_0_2.x * diff_0_2.x + diff_0_2.y * diff_0_2.y);

  if (dist_0_1 >= dist_1_2) {
    std::cout << "diff_0_1 >= diff_1_2" << std::endl;
  }
  if (dist_1_2 >= dist_0_2) {
    std::cout << "diff_1_2 >= diff_0_2" << std::endl;
  }
}

std::ostream & operator<<(std::ostream & os, const FrequencyCam::Event & e)
{
  os << std::fixed << std::setw(10) << std::setprecision(6) << e.t * 1e-6 << " "
     << static_cast<int>(e.polarity) << " " << e.x << " " << e.y;
  return (os);
}

}  // namespace frequency_cam
