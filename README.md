# Stress day prediction

This is a personal project focused on predicting daily stress events using machine learning. 
The goal is to analyze historical data and forecast "stress days", helping explore risk or stress patterns in the dataset.

## Setup

### clone repository:
```
git clone <repo-url>
cd Stress-day-prediction
```
### Create and activate a virtual environment
```
python -m venv venv
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```
### Install dependencies:
```
pip install -r requirements.txt
```
## Running the project
Theres one config file for both training and evaluation.
There's also one sample config file in configs/config.yaml
### training:
```
python scripts/train.py --config <path-to-config>
# can be a path from project root like confings/config.yaml
```
### Evaluation and PDF report:
```
python scripts/evaluate.py --config <path-to-config>
```
## Notebooks
Notebooks are intended for exploration and should not be relied on for reproducible runs - use the scripts for that.
Also currently they might not even work due to how I made the scripts work.
## Feedback and Suggestions
This is a personal project, so there’s no expectation for external contributions.
If you have suggestions, improvements, or ideas, feel free to open an issue or contact me.

