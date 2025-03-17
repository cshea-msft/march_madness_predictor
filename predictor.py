import os
import json
from typing import List, Tuple, Dict
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from datetime import datetime
from scraper import CBBStatsScraper
from historical_data import TournamentHistoryScraper
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
        self.history_scraper = TournamentHistoryScraper()
        self.seed_analyzer = SeedAnalyzer()
        
        print("Loading current season statistics...")
        self.team_stats = self._load_current_stats()
        
        print("Loading historical tournament data...")
        self.historical_data = self.history_scraper.get_historical_data()
        
        print("Analyzing seed performance...")
        self._initialize_seed_analysis()

    def _initialize_seed_analysis(self) -> None:
        """Initialize seed analysis from historical tournament data."""
        if not self.seed_analyzer.load_cache():
            self.seed_analyzer.analyze_tournament_data(self.historical_data['tournaments'])

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
        """Create a prompt for the model based on team statistics, historical performance, and seeds."""
        stats1 = self.team_stats.get(team1, {})
        stats2 = self.team_stats.get(team2, {})
        hist1 = self.historical_data['team_stats'].get(team1, {})
        hist2 = self.historical_data['team_stats'].get(team2, {})
        
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
        
        {team1} Historical Tournament Performance:
        - Tournament Appearances: {hist1.get('appearances', 0)}
        - Tournament Win Rate: {hist1.get('win_rate', 0):.3f}
        - Championships: {hist1.get('championships', 0)}
        - Final Fours: {hist1.get('final_fours', 0)}
        - Upset Win Rate: {hist1.get('upset_rate', 0):.3f}
        
        {team2} Current Season Stats:
        - Win Rate: {stats2.get('win_rate', 'N/A')}
        - Points Per Game: {stats2.get('points_per_game', 'N/A')}
        - Points Allowed: {stats2.get('points_allowed', 'N/A')}
        - Offensive Rating: {stats2.get('offensive_rating', 'N/A')}
        - Defensive Rating: {stats2.get('defensive_rating', 'N/A')}
        - Strength of Schedule: {stats2.get('sos', 'N/A')}
        - Simple Rating System: {stats2.get('srs', 'N/A')}
        
        {team2} Historical Tournament Performance:
        - Tournament Appearances: {hist2.get('appearances', 0)}
        - Tournament Win Rate: {hist2.get('win_rate', 0):.3f}
        - Championships: {hist2.get('championships', 0)}
        - Final Fours: {hist2.get('final_fours', 0)}
        - Upset Win Rate: {hist2.get('upset_rate', 0):.3f}
        
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
        
        # Get historical tournament performance
        hist1 = self.historical_data['team_stats'].get(team1, {})
        hist2 = self.historical_data['team_stats'].get(team2, {})
        
        # Calculate current season rating difference (30%)
        current_rating_diff = (
            (stats1.get('srs', 0) - stats2.get('srs', 0)) * 0.15 +  # SRS weight
            (stats1.get('win_rate', 0) - stats2.get('win_rate', 0)) * 0.1 +  # Win rate weight
            (stats1.get('offensive_rating', 0) - stats2.get('offensive_rating', 0)) * 0.025 +  # Offensive rating weight
            (stats2.get('defensive_rating', 0) - stats1.get('defensive_rating', 0)) * 0.025  # Defensive rating weight
        )
        
        # Calculate historical tournament performance difference (20%)
        historical_diff = (
            (hist1.get('win_rate', 0) - hist2.get('win_rate', 0)) * 0.08 +  # Tournament win rate
            (hist1.get('championship_rate', 0) - hist2.get('championship_rate', 0)) * 0.05 +  # Championship success
            (hist1.get('final_four_rate', 0) - hist2.get('final_four_rate', 0)) * 0.04 +  # Final Four success
            (hist1.get('upset_rate', 0) - hist2.get('upset_rate', 0)) * 0.03  # Upset factor
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
        total_rating_diff = current_rating_diff + historical_diff + seed_diff
        
        # Convert rating difference to win probability using sigmoid function
        win_prob = 1 / (1 + torch.exp(-torch.tensor(total_rating_diff)))
        
        # Determine winner and confidence
        winner = team1 if win_prob > 0.5 else team2
        confidence = float(win_prob if winner == team1 else 1 - win_prob)
        
        return winner, confidence

    def predict_bracket(self, bracket_teams: List[List[str]], seeds: List[int] = None) -> List[str]:
        """
        Predict the entire tournament bracket.
        bracket_teams should be a list of lists, representing each round's matchups.
        seeds should be a list of seeds corresponding to bracket_teams.
        """
        results = []
        current_round = bracket_teams
        current_seeds = seeds if seeds else [None] * len(bracket_teams)
        
        while len(current_round) > 1:
            next_round = []
            next_seeds = []
            round_name = self._get_round_name(len(current_round))
            results.append(f"\n{round_name}:")
            
            for i in range(0, len(current_round), 2):
                team1, team2 = current_round[i], current_round[i + 1]
                seed1 = current_seeds[i] if current_seeds else None
                seed2 = current_seeds[i + 1] if current_seeds else None
                
                winner, confidence = self.predict_game(team1, team2, seed1, seed2, round_name)
                next_round.append(winner)
                next_seeds.append(seed1 if winner == team1 else seed2)
                
                # Add prediction context
                matchup_note = self._get_matchup_note(team1, team2, seed1, seed2, round_name)
                results.append(f"{team1} ({seed1 if seed1 else 'N/A'}) vs {team2} ({seed2 if seed2 else 'N/A'}): {winner} wins ({confidence:.1%} confidence)")
                if matchup_note:
                    results.append(f"Note: {matchup_note}")
            
            current_round = next_round
            current_seeds = next_seeds
        
        if current_round:
            results.append("\nNational Champion:")
            champion = current_round[0]
            champion_seed = current_seeds[0] if current_seeds else None
            hist_champ = self.historical_data['team_stats'].get(champion, {})
            seed_stats = self.seed_analyzer.get_seed_metrics(champion_seed) if champion_seed else {}
            
            results.append(f"{champion} (#{champion_seed if champion_seed else 'N/A'}) 🏆")
            results.append(f"Previous Championships: {hist_champ.get('championships', 0)}")
            if champion_seed:
                results.append(f"Seed #{champion_seed} Championship Rate: {seed_stats.get('championship_rate', 0):.1%}")
        
        return results

    def _get_matchup_note(self, team1: str, team2: str, seed1: int = None, seed2: int = None, round_name: str = None) -> str:
        """Generate a note about the matchup including historical and seed-based context."""
        notes = []
        
        # Add historical tournament context
        hist_note = self._get_historical_matchup_note(team1, team2, round_name)
        if hist_note:
            notes.append(hist_note)
        
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

    def _get_historical_matchup_note(self, team1: str, team2: str, round_name: str) -> str:
        """Generate a note about historical tournament performance for this matchup."""
        hist1 = self.historical_data['team_stats'].get(team1, {})
        hist2 = self.historical_data['team_stats'].get(team2, {})
        
        notes = []
        
        # Add championship history if in later rounds
        if round_name in ["Championship Game", "Final Four", "Elite Eight"]:
            if hist1.get('championships', 0) > 0:
                notes.append(f"{team1} has {hist1['championships']} previous championships")
            if hist2.get('championships', 0) > 0:
                notes.append(f"{team2} has {hist2['championships']} previous championships")
        
        # Add upset potential
        if hist1.get('upset_rate', 0) > 0.3:
            notes.append(f"{team1} has strong upset history ({hist1['upset_rate']:.1%} upset rate)")
        if hist2.get('upset_rate', 0) > 0.3:
            notes.append(f"{team2} has strong upset history ({hist2['upset_rate']:.1%} upset rate)")
        
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