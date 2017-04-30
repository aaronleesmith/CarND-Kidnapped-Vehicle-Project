/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

#define NUM_PARTICLES 500;

#pragma clean diagnostic push
#pragma ide diagnostic ignored "IncompatibleTypes"

/**
 * Initialize particles around gaussian and weights to 1.
 * @param x GPS Pos X
 * @param y GPS Pos Y
 * @param theta Heading Est
 * @param std Uncertainties
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = NUM_PARTICLES;
	std::default_random_engine gen;
	double std_x = std[0], std_y = std[1], std_theta = std[2];

	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	// Create particles & initialize all weights to 1.
	for (int i = 0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle p = {i, sample_x, sample_y, sample_theta, 1.0};
		particles.push_back(p);
    weights.push_back(p.weight);
	}

  is_initialized = true;
}

/**
 * Updates each particle's position estimates and accounts for sensor noise using std_pos (mean is the particles current position).
 * @param delta_t Amount of time between time steps.
 * @param std_pos Velocity and yaw rate uncertainties.
 * @param velocity Velocity measurement
 * @param yaw_rate Yaw measurement
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  std::default_random_engine gen;

	for(int i = 0; i < num_particles; i++) {
    double d_x, d_y, d_theta = 0;
		double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2];

    if (yaw_rate == 0) {
      d_theta = 0;
      d_x = velocity * delta_t * sin(particles[i].theta);
      d_y = velocity * delta_t * cos(particles[i].theta);
    } else {
      d_theta = yaw_rate * delta_t;
      d_x = (velocity / yaw_rate) *  (sin(particles[i].theta + d_theta) - sin(particles[i].theta));
      d_y = (velocity / yaw_rate) *  (cos(particles[i].theta ) - cos(particles[i].theta + d_theta));
    }

		std::normal_distribution<double> dist_x(d_x, std_x);
		std::normal_distribution<double> dist_y(d_y, std_y);
		std::normal_distribution<double> dist_theta(d_theta, std_theta);

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
	}
}

/**
 * Goal: Perform NN and assign each sensor obs. to map landmark ID associated with it.
 * @param predicted Predicted measurements between one particle and all landmarks in sensor range.
 * @param observations Actual landmark measurements from lidar.
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  for(int i = 0; i < observations.size(); i++) {
    int min_id;
    double min_distance = INFINITY;
    LandmarkObs observation = observations[i];

    for (int j = 0; j < predicted.size(); j++) {
      double distance = dist(observation.x, observation.y, predicted[j].x, predicted[j].y);
      if (distance < min_distance) {
        min_distance = distance;
        min_id = predicted[j].id;
      }
    }

    observations[i].id = min_id;
  }
}

/**
 * Predict measurements to all the map landmarks within sensor range to this particle. Then use data association
 * to associate sensor measurements to map landmarks.
 *
 * Then, use MVG to update weights and normalize.
 * @param sensor_range Range of sensor
 * @param std_landmark Uncertainties
 * @param observations Landmark measurements
 * @param map_landmarks Map landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  for(int i = 0; i < num_particles; i++) {
    std::vector<LandmarkObs> transformed_observations;
    std::vector<LandmarkObs> landmarks_in_range;
    double  p_theta = particles[i].theta,
            p_x = particles[i].x,
            p_y = particles[i].y;

    // Transform all the observations into the coordinate space of this particle.
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs observation = observations[j];
      LandmarkObs transformed_observation;

      transformed_observation.x = observation.x * cos(p_theta) - observation.y * sin(p_theta) + p_x;
      transformed_observation.y = observation.x * sin(p_theta) + observation.y * cos(p_theta) + p_y;
      transformed_observations.push_back(transformed_observation);
    }

    // Get all the landmarks in range of this particle.
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      if (dist(p_x, p_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) < sensor_range) {
        LandmarkObs l;
        l.x = map_landmarks.landmark_list[j].x_f;
        l.y = map_landmarks.landmark_list[j].y_f;
        l.id = map_landmarks.landmark_list[j].id_i;
        landmarks_in_range.push_back(l);
      }
    }

    // Perform nearest neighbor (data association to get the closest landmark ID for each observation).
    dataAssociation(landmarks_in_range, transformed_observations);

    double weight = 1.0;
    for (int j = 0; j < transformed_observations.size(); j++) {
      LandmarkObs observation = transformed_observations[j];
      LandmarkObs landmark;

      try {
        landmark = findLandmarkById(observation.id, landmarks_in_range);
      } catch(std::exception& e) {
        continue;
      }

      weight *= exp(-0.5 * (pow(landmark.x - observation.x, 2.0) * std_landmark[0] + pow(landmark.y - observation.y, 2.0) * std_landmark[1])) / sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    }

    weights[i] = weight;
    particles[i].weight = weight;

  }
}

LandmarkObs ParticleFilter::findLandmarkById(int id, std::vector<LandmarkObs> landmarks) {
  for(int i = 0; i < landmarks.size(); i++) {
    if (landmarks[i].id == id) {
      return landmarks[i];
    }
  }

  throw std::exception();
}

/**
 * Update particles to postierior.
 */
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> resampled_particles;
  std::discrete_distribution<int> weight_distribution(weights.begin(), weights.end());
  std::default_random_engine gen;

  for(int i = 0; i < particles.size(); i++) {
    resampled_particles.push_back(particles[weight_distribution(gen)]);
  }

  particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
