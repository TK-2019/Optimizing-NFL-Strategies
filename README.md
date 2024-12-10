# Optimizing-NFL-Strategies
This repository contains the project Optimizing NFL Strategies, developed as part of the course MTH443A - Statistical and AI Techniques in Data Mining at the Indian Institute of Technology Kanpur. The project applies advanced data mining and machine learning techniques to analyze NFL game data and provide actionable insights into play strategies and outcomes.

# Objectives
## 1. Pass Outcome Prediction -
1. Predict the success of passes using game and player features.
2. Identify factors influencing successful pass outcomes.

## 2. Feature Engineering - 
Create derived metrics (e.g., catch separation, receiver-mate distance) for better model predictions.

## 3. Insights into Optimal Play - 
Explore correlations between game strategies and success rates to recommend best practices.

# Dataset
Source: NFL Big Data Bowl 2024 - https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data

# Methodology
## 1. Feature Engineering - 
Several self-engineered features were created for analysis and prediction:
1. Catch Separation: Distance between the receiver and the closest defender during a catch.
2. Receiver-Mate Distance: Distance between the receiver and other teammates at the time of the catch.
3. Last Catch Yardage: Distance covered after a catch until a tackle or other event.
4. Pass Outcome: Derived from play data to indicate success (1) or failure (0) based on specific criteria.

The code and the outputs are uploaded in the features folder.

## 2. Pass Outcome Prediction - 
1. Using Logistic Regression
2. Using Random Forest

The code is uploaded in models folder.

## 3. Extra Insights - 
1. Optimal Receive Locations: Heatmaps indicate high-success zones for receiving passes.
2. Impact of Catch Separation: Greater separation increases the likelihood of successful passes.
3. Offense Formation Analysis: Radar charts highlight formations with higher success rates.

The code for same is uploaded in insights folder.

The full project report is also uploaded for reference.


