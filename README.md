# March Madness Predictor

A machine learning-powered predictor for NCAA March Madness basketball tournament outcomes.

## Features

- **Team Statistics Scraping**: Automatically collects current season statistics for NCAA basketball teams.
- **Seed-Based Analysis**: Uses historical seed performance data to inform predictions.
- **Comprehensive Predictions**: Generates detailed matchup predictions with confidence scores.
- **Tournament Simulation**: Simulates the entire tournament bracket from First Round to Championship.
- **Upset Detection**: Identifies potential upsets based on seed matchup history.

## How It Works

The March Madness Predictor uses a combination of current season team statistics and historical seed performance data to generate predictions:

1. **Current Season Data**: Team win rate, points per game, defensive/offensive ratings, strength of schedule, and other metrics for the current season are scraped from sports statistics websites.

2. **Seed Analysis**: Historical performance of seeds (1-16) in the tournament is analyzed to determine:
   - Typical win rates for each seed
   - Head-to-head performance between different seeds
   - Round-specific performance for seeds
   - Common upset patterns

3. **Prediction Algorithm**: Combines current season statistics (50%) and seed-based historical data (50%) to calculate win probabilities for each matchup.

## Setup and Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/march_madness_predictor.git
   cd march_madness_predictor
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Prediction

Run the main prediction script to generate a complete tournament bracket prediction:

```
python run_predictions.py
```

This will output predictions for each round of the tournament, from First Round to the National Championship.

### Example Output

```
=== SOUTH REGION ===

First Round:
Auburn (1) vs Grambling (16): Auburn wins (99.8% confidence)
Louisville (8) vs Creighton (9): Creighton wins (51.2% confidence)
Michigan (5) vs UC San Diego (12): Michigan wins (77.4% confidence)
Texas A&M (4) vs Yale (13): Texas A&M wins (85.1% confidence)
...

Sweet Sixteen:
Auburn (1) vs Creighton (9): Auburn wins (76.5% confidence)
Texas A&M (4) vs Michigan (5): Texas A&M wins (62.3% confidence)
...

Elite Eight:
Auburn (1) vs Texas A&M (4): Auburn wins (68.9% confidence) - #1 seeds have historically performed well in the Elite Eight (72.0% win rate)

=== FINAL FOUR ===

Semifinal 1: Auburn (1) vs Arizona (4): Auburn wins (60.5% confidence)
Semifinal 2: Kansas (7) vs Houston (1): Houston wins (72.3% confidence)

=== CHAMPIONSHIP ===

Championship: Auburn (1) vs Houston (1): Houston wins (65.9% confidence)

CHAMPION: Houston (1) 🏆
```

## Project Structure

- `run_predictions.py`: Main script for running the predictor
- `predictor.py`: Core prediction algorithm and model
- `scraper.py`: Scrapes current season team statistics
- `seed_analysis.py`: Analyzes historical seed performance

## Customization

### Using Different Teams

You can modify the team list in `run_predictions.py` to use different teams or tournament brackets:

```python
# Example for customizing teams in run_predictions.py
bracket_teams = [
    "Kentucky", "Duke", "North Carolina", "Kansas",
    "Gonzaga", "Villanova", "Michigan", "Texas"
]
seeds = [1, 2, 3, 4, 5, 6, 7, 8]

results = predictor.predict_bracket(bracket_teams, seeds)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sports Reference for team statistics data
- NCAA for tournament history and bracket information