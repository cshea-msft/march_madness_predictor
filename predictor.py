import os
import json
from typing import List, Tuple, Dict
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from datetime import datetime
from scraper import CBBStatsScraper
from seed_analysis import SeedAnalyzer

class MarchMadnessPredictor:
    def __init__(self):
        # Initialize the transformer model for text classification
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize the scrapers and analyzers
        self.stats_scraper = CBBStatsScraper()
        self.seed_analyzer = SeedAnalyzer()
        
        print("Loading current season statistics...")
        self.team_stats = self._load_current_stats()
        
        print("Analyzing seed performance...")
        self._initialize_seed_analysis()

    def _initialize_seed_analysis(self) -> None:
        """Initialize seed analysis using cached seed data."""
        self.seed_analyzer.load_cache()

    def _load_current_stats(self) -> Dict:
        """Load current team statistics using the scraper."""
        cached_stats = self.stats_scraper.load_cached_stats()
        
        if not cached_stats or (datetime.now() - datetime.fromisoformat(cached_stats.get('last_updated', '2000-01-01'))).days >= 1:
            print("Fetching fresh team statistics...")
            return self.stats_scraper.scrape_all_teams()
        else:
            print("Using cached team statistics...")
            return cached_stats

    def _create_game_prompt(self, team1: str, team2: str, seed1: int = None, seed2: int = None) -> str:
        """Create a prompt for the model based on team statistics and seeds."""
        stats1 = self.team_stats.get(team1, {})
        stats2 = self.team_stats.get(team2, {})
        
        seed_info = ""
        if seed1 is not None and seed2 is not None:
            seed_metrics1 = self.seed_analyzer.get_seed_metrics(seed1)
            seed_metrics2 = self.seed_analyzer.get_seed_metrics(seed2)
            
            seed_info = f"""
            Seed Information:
            {team1} (#{seed1}):
            - Historical Seed Win Rate: {seed_metrics1.get('win_rate', 0):.3f}
            - Seed Championship Rate: {seed_metrics1.get('championship_rate', 0):.3f}
            - Seed Final Four Rate: {seed_metrics1.get('final_four_rate', 0):.3f}
            
            {team2} (#{seed2}):
            - Historical Seed Win Rate: {seed_metrics2.get('win_rate', 0):.3f}
            - Seed Championship Rate: {seed_metrics2.get('championship_rate', 0):.3f}
            - Seed Final Four Rate: {seed_metrics2.get('final_four_rate', 0):.3f}
            """
        
        prompt = f"""
        Predict the winner of this March Madness matchup:
        {team1} vs {team2}
        
        {team1} Current Season Stats:
        - Win Rate: {stats1.get('win_rate', 'N/A')}
        - Points Per Game: {stats1.get('points_per_game', 'N/A')}
        - Points Allowed: {stats1.get('points_allowed', 'N/A')}
        - Offensive Rating: {stats1.get('offensive_rating', 'N/A')}
        - Defensive Rating: {stats1.get('defensive_rating', 'N/A')}
        - Strength of Schedule: {stats1.get('sos', 'N/A')}
        - Simple Rating System: {stats1.get('srs', 'N/A')}
        
        {team2} Current Season Stats:
        - Win Rate: {stats2.get('win_rate', 'N/A')}
        - Points Per Game: {stats2.get('points_per_game', 'N/A')}
        - Points Allowed: {stats2.get('points_allowed', 'N/A')}
        - Offensive Rating: {stats2.get('offensive_rating', 'N/A')}
        - Defensive Rating: {stats2.get('defensive_rating', 'N/A')}
        - Strength of Schedule: {stats2.get('sos', 'N/A')}
        - Simple Rating System: {stats2.get('srs', 'N/A')}
        
        {seed_info}
        """
        return prompt

    def predict_game(self, team1: str, team2: str, seed1: int = None, seed2: int = None, round_name: str = None) -> Tuple[str, float]:
        """
        Predict the winner of a game between two teams.
        Returns the predicted winner and the confidence score.
        """
        prompt = self._create_game_prompt(team1, team2, seed1, seed2)
        prediction = self.classifier(prompt)[0]
        
        # Get current season stats
        stats1 = self.team_stats.get(team1, {})
        stats2 = self.team_stats.get(team2, {})
        
        # Calculate current season rating difference (50%)
        current_rating_diff = (
            (stats1.get('srs', 0) - stats2.get('srs', 0)) * 0.25 +  # SRS weight
            (stats1.get('win_rate', 0) - stats2.get('win_rate', 0)) * 0.15 +  # Win rate weight
            (stats1.get('offensive_rating', 0) - stats2.get('offensive_rating', 0)) * 0.05 +  # Offensive rating weight
            (stats2.get('defensive_rating', 0) - stats1.get('defensive_rating', 0)) * 0.05  # Defensive rating weight
        )
        
        # Calculate seed-based probability difference (50%)
        seed_diff = 0
        if seed1 is not None and seed2 is not None:
            # Direct matchup probability (30%)
            seed_matchup_prob = self.seed_analyzer.get_seed_win_probability(seed1, seed2)
            seed_diff += (seed_matchup_prob - 0.5) * 0.3
            
            # Round-specific performance (10%)
            if round_name:
                round_prob1 = self.seed_analyzer.get_seed_round_probability(seed1, round_name)
                round_prob2 = self.seed_analyzer.get_seed_round_probability(seed2, round_name)
                seed_diff += (round_prob1 - round_prob2) * 0.1
            
            # Upset potential (10%)
            if seed1 > seed2:  # team1 would be the underdog
                upset_prob = self.seed_analyzer.get_upset_probability(seed1, seed2)
                seed_diff += (upset_prob - 0.5) * 0.1
            elif seed2 > seed1:  # team2 would be the underdog
                upset_prob = self.seed_analyzer.get_upset_probability(seed2, seed1)
                seed_diff -= (upset_prob - 0.5) * 0.1
        
        # Combine all factors
        total_rating_diff = current_rating_diff + seed_diff
        
        # Convert rating difference to win probability using sigmoid function
        win_prob = 1 / (1 + torch.exp(-torch.tensor(total_rating_diff)))
        
        # Determine winner and confidence
        winner = team1 if win_prob > 0.5 else team2
        confidence = float(win_prob if winner == team1 else 1 - win_prob)
        
        return winner, confidence

    def predict_bracket(self, bracket_teams: List[str], seeds: List[int] = None) -> List[str]:
        """
        Predict the winners of all games in a tournament bracket.
        Returns a list of results for each round.
        """
        results = []
        
        # Initialize the current round teams and seeds
        current_round = bracket_teams.copy()
        current_seeds = seeds.copy() if seeds else [None] * len(bracket_teams)
        
        # Simulate each round
        while len(current_round) > 1:
            round_name = self._get_round_name(len(current_round))
            results.append(f"\n{round_name}:")
            
            next_round = []
            next_seeds = []
            
            # Process each matchup in the current round
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    team1 = current_round[i]
                    team2 = current_round[i + 1]
                    seed1 = current_seeds[i] if current_seeds else None
                    seed2 = current_seeds[i + 1] if current_seeds else None
                    
                    winner, confidence = self.predict_game(team1, team2, seed1, seed2, round_name)
                    
                    # Format the result with detailed matchup analysis
                    seed1_display = f"({seed1})" if seed1 is not None else ""
                    seed2_display = f"({seed2})" if seed2 is not None else ""
                    result_str = f"{team1} {seed1_display} vs {team2} {seed2_display}: {winner} wins ({confidence:.1%} confidence)"
                    
                    # Add any special notes about the matchup
                    matchup_note = self._get_matchup_note(team1, team2, seed1, seed2, round_name)
                    if matchup_note:
                        result_str += f" - {matchup_note}"
                    
                    results.append(result_str)
                    
                    # Add the winner to the next round
                    next_round.append(winner)
                    next_seeds.append(seed1 if winner == team1 else seed2)
                else:
                    # Odd number of teams, this team gets a bye
                    next_round.append(current_round[i])
                    if current_seeds:
                        next_seeds.append(current_seeds[i])
            
            # Update for the next round
            current_round = next_round
            current_seeds = next_seeds
        
        if current_round:
            results.append("\nNational Champion:")
            champion = current_round[0]
            champion_seed = current_seeds[0] if current_seeds else None
            seed_stats = self.seed_analyzer.get_seed_metrics(champion_seed) if champion_seed else {}
            
            results.append(f"{champion} (#{champion_seed if champion_seed else 'N/A'}) [CHAMPION]")
            if champion_seed:
                results.append(f"Seed #{champion_seed} Championship Rate: {seed_stats.get('championship_rate', 0):.1%}")
        
        return results

    def _get_matchup_note(self, team1: str, team2: str, seed1: int = None, seed2: int = None, round_name: str = None) -> str:
        """Generate a note about the matchup including seed-based context."""
        notes = []
        
        # Add seed-based context
        if seed1 is not None and seed2 is not None:
            # Check for potential upsets
            if seed1 > seed2:
                upset_prob = self.seed_analyzer.get_upset_probability(seed1, seed2)
                if upset_prob > 0.2:
                    notes.append(f"Upset Alert: #{seed1} seeds have a {upset_prob:.1%} historical chance of beating #{seed2} seeds")
            elif seed2 > seed1:
                upset_prob = self.seed_analyzer.get_upset_probability(seed2, seed1)
                if upset_prob > 0.2:
                    notes.append(f"Upset Alert: #{seed2} seeds have a {upset_prob:.1%} historical chance of beating #{seed1} seeds")
            
            # Add round-specific seed performance
            if round_name:
                seed1_round_prob = self.seed_analyzer.get_seed_round_probability(seed1, round_name)
                seed2_round_prob = self.seed_analyzer.get_seed_round_probability(seed2, round_name)
                
                if abs(seed1_round_prob - seed2_round_prob) > 0.1:
                    better_seed = seed1 if seed1_round_prob > seed2_round_prob else seed2
                    better_prob = max(seed1_round_prob, seed2_round_prob)
                    notes.append(f"#{better_seed} seeds have historically performed well in the {round_name} ({better_prob:.1%} win rate)")
        
        return " | ".join(notes) if notes else ""

    def _get_round_name(self, num_teams: int) -> str:
        """Get the name of the tournament round based on number of teams."""
        round_names = {
            64: "First Round",
            32: "Second Round",
            16: "Sweet Sixteen",
            8: "Elite Eight",
            4: "Final Four",
            2: "Championship Game"
        }
        return round_names.get(num_teams, f"Round of {num_teams}")

def main():
    predictor = MarchMadnessPredictor()
    
    # Example usage with some top teams and their seeds
    example_bracket = [
        "Duke", "Kentucky", "Kansas", "North Carolina",
        "Gonzaga", "Purdue", "Houston", "Arizona"
    ]
    example_seeds = [1, 4, 2, 3, 1, 2, 3, 4]  # Example seeds for the teams
    
    # Predict tournament games
    print("\n2025 March Madness Predictions:")
    print("================================")
    results = predictor.predict_bracket(example_bracket, example_seeds)
    for result in results:
        print(result)

if __name__ == "__main__":
    main() 