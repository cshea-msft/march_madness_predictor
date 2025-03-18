#!/usr/bin/env python
import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Tuple, Any
import logging
from predictor import MarchMadnessPredictor
from seed_analysis import SeedAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for a model in the ensemble."""
    
    def __init__(self, name: str, model_path: str = None, weight: float = 1.0):
        self.name = name
        self.model_path = model_path
        self.weight = weight

class EnsemblePredictor:
    """Ensemble predictor that combines multiple models for March Madness predictions."""
    
    def __init__(self, year: int = 2025, models: List[ModelConfig] = None):
        self.year = year
        self.predictors = []
        self.model_weights = []
        
        # Default models if none provided
        if models is None:
            models = [
                ModelConfig("Fine-tuned BERT", "fine_tuned_model", 1.5),
                ModelConfig("Base BERT", None, 1.0),
                ModelConfig("Seed Analysis", None, 0.5)
            ]
        
        # Initialize predictors
        for model_config in models:
            if model_config.name == "Seed Analysis":
                # Special case for seed analyzer
                self.seed_analyzer = SeedAnalyzer()
                self.seed_analyzer.load_cached_data()
                logger.info("Successfully loaded seed analysis from cache")
            elif model_config.name == "Fine-tuned BERT":
                # Load fine-tuned model
                logger.info(f"Loading model {model_config.name} from {model_config.model_path}")
                try:
                    predictor = MarchMadnessPredictor(
                        year=self.year,
                        fine_tuned_model_path=model_config.model_path
                    )
                    self.predictors.append(predictor)
                    self.model_weights.append(model_config.weight)
                except Exception as e:
                    logger.error(f"Error loading {model_config.name}: {str(e)}")
            else:
                # Load base model
                logger.info(f"Using default pre-trained model for {model_config.name}")
                predictor = MarchMadnessPredictor(
                    year=self.year,
                    fine_tuned_model_path=None
                )
                self.predictors.append(predictor)
                self.model_weights.append(model_config.weight)
        
        logger.info(f"\nInitializing March Madness Ensemble Predictor for {self.year}...\n")
    
    def predict_matchup(self, team1: str, team2: str, seed1: int, seed2: int, 
                         region: str = None, round_name: str = None) -> Tuple[float, str]:
        """
        Predict the winner of a matchup using the ensemble of models.
        
        Args:
            team1: Name of the first team
            team2: Name of the second team
            seed1: Seed of the first team
            seed2: Seed of the second team
            region: Tournament region
            round_name: Tournament round name
            
        Returns:
            Tuple of (confidence, predicted winner)
        """
        votes = []
        confidences = []
        ensemble_confidence = 0.0
        
        # Get predictions from all models
        for i, predictor in enumerate(self.predictors):
            confidence, winner = predictor.predict_matchup(
                team1, team2, seed1, seed2, region, round_name
            )
            
            # Convert to binary vote (1 for team1, 0 for team2)
            vote = 1 if winner == team1 else 0
            votes.append(vote)
            confidences.append(confidence)
            
            # Apply model weight
            ensemble_confidence += confidences[i] * self.model_weights[i] * (1 if vote else -1)
        
        # Get seed-based win probability
        seed_win_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2, round_name)
        
        # Add seed analyzer "vote" with its weight
        seed_vote = 1 if seed_win_prob >= 0.5 else 0
        votes.append(seed_vote)
        
        # Normalize ensemble confidence to [0, 1]
        sum_weights = sum(self.model_weights) + 0.5  # Add weight for seed analyzer
        ensemble_confidence = 0.5 + (ensemble_confidence / (2 * sum_weights))
        
        # Ensure confidence is in [0, 1]
        ensemble_confidence = max(0.01, min(0.99, ensemble_confidence))
        
        # Determine winner based on majority vote
        ensemble_votes_weighted = sum(v * w for v, w in zip(votes[:-1], self.model_weights))
        ensemble_votes_weighted += seed_vote * 0.5  # Add seed analyzer vote
        
        winner = team1 if ensemble_votes_weighted / sum_weights >= 0.5 else team2
        
        # Adjust confidence to reflect winning team
        if winner == team2:
            ensemble_confidence = 1 - ensemble_confidence
            
        return ensemble_confidence, winner
    
    def get_consensus_stats(self, team1: str, team2: str, seed1: int, seed2: int) -> Dict[str, float]:
        """
        Get statistics about the model consensus for a matchup.
        
        Args:
            team1: Name of the first team
            team2: Name of the second team
            seed1: Seed of the first team
            seed2: Seed of the second team
            
        Returns:
            Dictionary with consensus statistics
        """
        votes_team1 = 0
        votes_team2 = 0
        
        # Get votes from all models
        for predictor in self.predictors:
            _, winner = predictor.predict_matchup(team1, team2, seed1, seed2)
            if winner == team1:
                votes_team1 += 1
            else:
                votes_team2 += 1
        
        # Get seed-based prediction
        seed_win_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2)
        seed_vote = team1 if seed_win_prob >= 0.5 else team2
        
        if seed_vote == team1:
            votes_team1 += 1
        else:
            votes_team2 += 1
        
        # Calculate percentages
        total_votes = votes_team1 + votes_team2
        team1_percentage = (votes_team1 / total_votes) * 100
        team2_percentage = (votes_team2 / total_votes) * 100
        
        return {
            team1: team1_percentage,
            team2: team2_percentage,
            "consensus": team1 if votes_team1 > votes_team2 else team2
        }
    
    def generate_bracket(self, regions: List[str] = None, save_details: bool = False) -> Dict[str, Any]:
        """
        Generate complete bracket predictions for the tournament.
        
        Args:
            regions: List of regions
            save_details: Whether to save detailed prediction info
            
        Returns:
            Dictionary with predicted bracket
        """
        if regions is None:
            regions = ["South", "East", "West", "Midwest"]
            
        # Dictionary to store results
        bracket = {
            "regions": {},
            "final_four": {},
            "championship": {},
            "champion": None
        }
        
        # Teams that advance from each region
        region_winners = {}
        
        # Process each region
        for region in regions:
            logger.info(f"\n=== {region.upper()} REGION (ENSEMBLE PREDICTION) ===\n")
            bracket["regions"][region] = self._predict_region(region, save_details)
            
            # The region winner is the team that wins the regional final
            region_winners[region] = bracket["regions"][region]["elite_eight"][0]["winner"]
        
        # Final Four (national semifinals)
        logger.info("\n=== FINAL FOUR (ENSEMBLE PREDICTION) ===\n")
        bracket["final_four"] = self._predict_final_four(region_winners, save_details)
        
        # Championship game
        finalists = [m["winner"] for m in bracket["final_four"]]
        logger.info("\n=== CHAMPIONSHIP GAME (ENSEMBLE PREDICTION) ===\n")
        bracket["championship"] = self._predict_championship(finalists[0], finalists[1], save_details)
        
        # National champion
        bracket["champion"] = bracket["championship"]["winner"]
        
        logger.info(f"\nNational Champion:\n{bracket['champion']} [CHAMPION]")
        
        return bracket
    
    def _predict_region(self, region: str, save_details: bool = False) -> Dict[str, List[Dict]]:
        """
        Predict all games in a region.
        
        Args:
            region: Region name
            save_details: Whether to save detailed prediction info
            
        Returns:
            Dictionary with predicted matchups for each round
        """
        # First round matchups (1 vs 16, 8 vs 9, etc.)
        first_round = []
        
        # Standard first round matchups
        matchups = [
            (1, 16), (8, 9), (5, 12), (4, 13),
            (6, 11), (3, 14), (7, 10), (2, 15)
        ]
        
        # Example team names based on seed
        # In a real implementation, these would come from actual tournament data
        teams = {
            1: "Auburn",
            2: "Michigan State",
            3: "Iowa State",
            4: "Texas A&M",
            5: "Michigan",
            6: "Ole Miss",
            7: "Marquette",
            8: "Louisville",
            9: "Creighton",
            10: "New Mexico",
            11: "Virginia",
            12: "UC San Diego",
            13: "Yale",
            14: "Lipscomb",
            15: "Bryant",
            16: "Grambling"
        }
        
        # Generate first round matchups with predictions
        logger.info("Sweet Sixteen:")
        for seed1, seed2 in matchups:
            team1 = f"{teams[seed1]}"
            team2 = f"{teams[seed2]}"
            
            # Predict winner
            confidence, winner = self.predict_matchup(
                team1, team2, seed1, seed2,
                region=region, round_name="Sweet Sixteen"
            )
            
            # Get consensus statistics
            consensus = self.get_consensus_stats(team1, team2, seed1, seed2)
            
            # Get seed matchup history
            seed_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2)
            seed_history = self.seed_analyzer.get_matchup_history(seed1, seed2)
            
            # Generate insight based on seed history
            insight = ""
            if seed1 < seed2 and seed_prob < 0.7:
                insight = f"Upset Alert: #{seed2} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed1} seeds"
            elif seed1 > seed2 and seed_prob > 0.3:
                insight = f"Upset Alert: #{seed1} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed2} seeds"
            
            # Add seed performance in this round
            round_performance = self.seed_analyzer.get_seed_round_performance(min(seed1, seed2))
            if round_performance > 0.25:
                if insight:
                    insight += " | "
                insight += f"#{min(seed1, seed2)} seeds have historically performed well in the Sweet Sixteen ({round_performance*100:.1f}% win rate)"
            
            # Log prediction
            logger.info(f"{team1} ({seed1}) vs {team2} ({seed2}): {winner} wins ({confidence*100:.1f}% confidence) [Ensemble consensus: {team1}: {consensus[team1]:.1f}%, {team2}: {consensus[team2]:.1f}%] - {insight}")
            
            matchup = {
                "team1": team1,
                "team2": team2,
                "seed1": seed1,
                "seed2": seed2,
                "winner": winner,
                "confidence": confidence,
                "region": region,
                "round": "Sweet Sixteen"
            }
            
            if save_details:
                matchup["consensus"] = consensus
                matchup["seed_probability"] = seed_prob
                matchup["seed_history"] = seed_history
                matchup["insight"] = insight
            
            first_round.append(matchup)
        
        # Generate second round (Elite Eight) matchups based on first round winners
        logger.info("\nElite Eight:")
        second_round = []
        
        for i in range(0, len(first_round), 2):
            winner1 = first_round[i]["winner"]
            winner2 = first_round[i+1]["winner"]
            
            # Get seeds
            seed1 = first_round[i]["seed1"] if first_round[i]["winner"] == first_round[i]["team1"] else first_round[i]["seed2"]
            seed2 = first_round[i+1]["seed1"] if first_round[i+1]["winner"] == first_round[i+1]["team1"] else first_round[i+1]["seed2"]
            
            # Predict winner
            confidence, winner = self.predict_matchup(
                winner1, winner2, seed1, seed2,
                region=region, round_name="Elite Eight"
            )
            
            # Get consensus statistics
            consensus = self.get_consensus_stats(winner1, winner2, seed1, seed2)
            
            # Get seed matchup history
            seed_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2, "Elite Eight")
            seed_history = self.seed_analyzer.get_matchup_history(seed1, seed2, "Elite Eight")
            
            # Generate insight based on seed history
            insight = ""
            if seed1 < seed2 and seed_prob < 0.7:
                insight = f"Upset Alert: #{seed2} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed1} seeds"
            elif seed1 > seed2 and seed_prob > 0.3:
                insight = f"Upset Alert: #{seed1} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed2} seeds"
            
            # Add seed performance in this round
            round_performance = self.seed_analyzer.get_seed_round_performance(min(seed1, seed2), "Elite Eight")
            if round_performance > 0.25:
                if insight:
                    insight += " | "
                insight += f"#{min(seed1, seed2)} seeds have historically performed well in the Elite Eight ({round_performance*100:.1f}% win rate)"
            
            # Log prediction
            logger.info(f"{winner1} ({seed1}) vs {winner2} ({seed2}): {winner} wins ({confidence*100:.1f}% confidence) [Ensemble consensus: {winner1}: {consensus[winner1]:.1f}%, {winner2}: {consensus[winner2]:.1f}%] - {insight}")
            
            matchup = {
                "team1": winner1,
                "team2": winner2,
                "seed1": seed1,
                "seed2": seed2,
                "winner": winner,
                "confidence": confidence,
                "region": region,
                "round": "Elite Eight"
            }
            
            if save_details:
                matchup["consensus"] = consensus
                matchup["seed_probability"] = seed_prob
                matchup["seed_history"] = seed_history
                matchup["insight"] = insight
            
            second_round.append(matchup)
        
        # Generate Sweet 16 matchups based on Elite Eight winners
        logger.info("\nFinal Four:")
        sweet_16 = []
        
        # Only one game in Sweet 16 per region
        winner1 = second_round[0]["winner"]
        winner2 = second_round[1]["winner"]
        
        # Get seeds
        seed1 = second_round[0]["seed1"] if second_round[0]["winner"] == second_round[0]["team1"] else second_round[0]["seed2"]
        seed2 = second_round[1]["seed1"] if second_round[1]["winner"] == second_round[1]["team1"] else second_round[1]["seed2"]
        
        # Predict winner
        confidence, winner = self.predict_matchup(
            winner1, winner2, seed1, seed2,
            region=region, round_name="Final Four"
        )
        
        # Get consensus statistics
        consensus = self.get_consensus_stats(winner1, winner2, seed1, seed2)
        
        # Get seed matchup history
        seed_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2, "Final Four")
        
        # Generate insight based on seed history
        insight = ""
        if seed1 < seed2 and seed_prob < 0.7:
            insight = f"Upset Alert: #{seed2} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed1} seeds"
        elif seed1 > seed2 and seed_prob > 0.3:
            insight = f"Upset Alert: #{seed1} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed2} seeds"
        
        # Log prediction
        logger.info(f"{winner1} ({seed1}) vs {winner2} ({seed2}): {winner} wins ({confidence*100:.1f}% confidence) [Ensemble consensus: {winner1}: {consensus[winner1]:.1f}%, {winner2}: {consensus[winner2]:.1f}%] - {insight}")
        
        matchup = {
            "team1": winner1,
            "team2": winner2,
            "seed1": seed1,
            "seed2": seed2,
            "winner": winner,
            "confidence": confidence,
            "region": region,
            "round": "Final Four"
        }
        
        if save_details:
            matchup["consensus"] = consensus
            matchup["seed_probability"] = seed_prob
            matchup["insight"] = insight
        
        sweet_16.append(matchup)
        
        return {
            "sweet_sixteen": first_round,
            "elite_eight": second_round,
            "final_four": sweet_16
        }
    
    def _predict_final_four(self, region_winners: Dict[str, str], save_details: bool = False) -> List[Dict]:
        """
        Predict the Final Four (national semifinals).
        
        Args:
            region_winners: Dictionary mapping region names to winners
            save_details: Whether to save detailed prediction info
            
        Returns:
            List of predicted matchups
        """
        # Standard Final Four matchups
        # South vs East, West vs Midwest
        matchups = [
            ("South", "East"),
            ("West", "Midwest")
        ]
        
        results = []
        
        for region1, region2 in matchups:
            team1 = region_winners[region1]
            team2 = region_winners[region2]
            
            # Determine seeds
            # In a real implementation, these would be looked up
            # Here we'll use placeholders based on previous predictions
            seed1 = 1 if team1 == "Auburn" else 2 if team1 == "Michigan State" else 3 if team1 == "Iowa State" else 4
            seed2 = 1 if team2 == "Auburn" else 2 if team2 == "Michigan State" else 3 if team2 == "Iowa State" else 4
            
            # Predict winner
            confidence, winner = self.predict_matchup(
                team1, team2, seed1, seed2,
                round_name="Final Four"
            )
            
            # Get consensus statistics
            consensus = self.get_consensus_stats(team1, team2, seed1, seed2)
            
            # Get seed matchup history
            seed_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2, "Final Four")
            
            # Generate insight based on seed history
            insight = ""
            if seed1 < seed2 and seed_prob < 0.7:
                insight = f"Upset Alert: #{seed2} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed1} seeds"
            elif seed1 > seed2 and seed_prob > 0.3:
                insight = f"Upset Alert: #{seed1} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed2} seeds"
            
            # Log prediction
            logger.info(f"{team1} ({seed1}) vs {team2} ({seed2}): {winner} wins ({confidence*100:.1f}% confidence) [Ensemble consensus: {team1}: {consensus[team1]:.1f}%, {team2}: {consensus[team2]:.1f}%] - {insight}")
            
            matchup = {
                "team1": team1,
                "team2": team2,
                "seed1": seed1,
                "seed2": seed2,
                "winner": winner,
                "confidence": confidence,
                "region1": region1,
                "region2": region2,
                "round": "Final Four"
            }
            
            if save_details:
                matchup["consensus"] = consensus
                matchup["seed_probability"] = seed_prob
                matchup["insight"] = insight
            
            results.append(matchup)
            
        return results
    
    def _predict_championship(self, team1: str, team2: str, save_details: bool = False) -> Dict:
        """
        Predict the championship game.
        
        Args:
            team1: First finalist
            team2: Second finalist
            save_details: Whether to save detailed prediction info
            
        Returns:
            Dictionary with championship prediction
        """
        # Determine seeds
        # In a real implementation, these would be looked up
        # Here we'll use placeholders based on previous predictions
        seed1 = 1 if team1 == "Auburn" else 2 if team1 == "Michigan State" else 3 if team1 == "Iowa State" else 4
        seed2 = 1 if team2 == "Auburn" else 2 if team2 == "Michigan State" else 3 if team2 == "Iowa State" else 4
        
        # Predict winner
        confidence, winner = self.predict_matchup(
            team1, team2, seed1, seed2,
            round_name="Championship"
        )
        
        # Get consensus statistics
        consensus = self.get_consensus_stats(team1, team2, seed1, seed2)
        
        # Get seed matchup history
        seed_prob = self.seed_analyzer.get_matchup_win_probability(seed1, seed2, "Championship")
        
        # Generate insight based on seed history
        insight = ""
        if seed1 < seed2 and seed_prob < 0.7:
            insight = f"Upset Alert: #{seed2} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed1} seeds"
        elif seed1 > seed2 and seed_prob > 0.3:
            insight = f"Upset Alert: #{seed1} seeds have a {seed_prob*100:.1f}% historical chance of beating #{seed2} seeds"
        
        # Log prediction
        logger.info(f"\nChampionship Game:")
        logger.info(f"{team1} ({seed1}) vs {team2} ({seed2}): {winner} wins ({confidence*100:.1f}% confidence) [Ensemble consensus: {team1}: {consensus[team1]:.1f}%, {team2}: {consensus[team2]:.1f}%] - {insight}")
        
        # Add championship rate statistic
        champ_seed = seed1 if winner == team1 else seed2
        champ_rate = self.seed_analyzer.get_seed_championship_rate(champ_seed)
        logger.info(f"Seed #{champ_seed} Championship Rate: {champ_rate*100:.1f}%")
        
        result = {
            "team1": team1,
            "team2": team2,
            "seed1": seed1,
            "seed2": seed2,
            "winner": winner,
            "confidence": confidence,
            "round": "Championship"
        }
        
        if save_details:
            result["consensus"] = consensus
            result["seed_probability"] = seed_prob
            result["insight"] = insight
            result["champion_seed"] = champ_seed
            result["champion_seed_rate"] = champ_rate
        
        return result

def main():
    parser = argparse.ArgumentParser(description="March Madness Ensemble Predictor")
    parser.add_argument("--year", type=int, default=2025,
                      help="Tournament year")
    parser.add_argument("--regions", type=str, default="South,East,West,Midwest",
                      help="Comma-separated list of regions")
    parser.add_argument("--save-details", action="store_true",
                      help="Save detailed prediction information")
    parser.add_argument("--output-file", type=str, default=None,
                      help="Output file path for predictions (JSON)")
    args = parser.parse_args()
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(year=args.year)
    
    # Generate bracket
    regions = args.regions.split(',')
    bracket = ensemble.generate_bracket(
        regions=regions,
        save_details=args.save_details
    )
    
    # Save predictions to file if specified
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(bracket, f, indent=2)
        logger.info(f"\nPredictions saved to {args.output_file}")

if __name__ == "__main__":
    main() 