# Chatbot for Wearable Technology Support

## Project Overview
This project involves the development of a chatbot designed to assist users of wearable technology with immediate and automated support. The chatbot is equipped to handle inquiries related health monitoring features, and emergency protocols.

## Features
- **Automated Responses**: Delivers instant responses to user queries 
- **Health Advice**: Provides guidance on health-related questions.
- **Emergency Support**: Offers quick assistance and instructions in case of detected emergencies.

## Installation
Follow these instructions to set up the chatbot on your local system.

### Prerequisites
Ensure you have Python 3.8 or higher installed, along with pip for managing packages.

### dataset
Data used for training: concatenated_data.csv

## model
This is the most advanced dialoggpt model available in the public library which uses transformers algorithm for its training. Eventough the model is created to chat in a casual manner, this program has customised the model to suit the requirements of the project.

### Creating the environment
If you are using Conda, the `environment.yml` file will help you create an environment with all required dependencies.
type these commands in command prompt:
conda env create -f environment.yml
conda activate chatbot-env
