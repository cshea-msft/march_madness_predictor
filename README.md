# March Madness Predictor

A machine learning-based predictor for NCAA March Madness basketball tournament outcomes.

## Overview

This project uses statistical analysis, historical tournament data, and machine learning to predict the outcomes of NCAA March Madness basketball tournament games. It scrapes current season statistics, analyzes historical tournament performance, and uses a transformer-based model to make predictions.

## Features

- **Team Statistics Scraper**: Collects current season statistics for all NCAA Division I teams
- **Historical Tournament Analysis**: Analyzes past tournament results to identify patterns
- **Seed Performance Analysis**: Evaluates how different seeds have performed historically
- **Machine Learning Prediction**: Uses a BERT-based model to predict game outcomes
- **Complete Bracket Prediction**: Generates predictions for the entire tournament bracket

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/cshea-msft/march_madness_predictor.git
   cd march_madness_predictor
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. Update team statistics (optional, will use cached data if available):
   ```
   python scraper.py
   ```

2. Run the predictor with the current tournament bracket:
   ```
   python run_predictions.py
   ```

## Components

- `scraper.py`: Scrapes current season team statistics
- `historical_data.py`: Handles historical tournament data
- `seed_analysis.py`: Analyzes seed performance in past tournaments
- `predictor.py`: Main prediction model
- `run_predictions.py`: Script to run predictions with the current bracket

## Data Sources

- Current season statistics: Sports Reference College Basketball
- Historical tournament data: NCAA tournament archives

## License

MIT

## Author

Chris Shea (@cshea-msft)

## How it Works

The predictor uses a transformer-based language model to analyze team statistics and predict game outcomes. It considers factors such as:
- Win rate
- Points per game
- Defense rating
- Strength of schedule

For best results, you should:
1. Update team statistics with current season data
2. Fine-tune the model with historical March Madness data
3. Input the correct tournament bracket structure

## Limitations

- The current version uses a base BERT model and would benefit from fine-tuning on historical basketball data
- Predictions are based on available statistics and may not account for real-time factors like injuries or recent form
- The model's accuracy depends on the quality and recency of the input data

## Future Improvements

- Add web scraping for real-time team statistics
- Implement fine-tuning on historical March Madness data
- Add support for probability distributions and upset predictions
- Include player-specific statistics and injury reports