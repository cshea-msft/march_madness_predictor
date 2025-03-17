import numpy as np
from typing import Dict, Tuple, List
import json
from datetime import datetime
import logging

class SeedAnalyzer:
    def __init__(self):
        self.cache_file = "seed_analysis_cache.json"
        self.seed_stats = {}
        self.matchup_stats = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_cache(self) -> bool:
        """Load seed analysis data from cache."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                
            self.seed_stats = data.get('seed_stats', {})
            self.matchup_stats = data.get('matchup_stats', {})
            
            # Convert string keys to integers for seed_stats
            self.seed_stats = {int(k): v for k, v in self.seed_stats.items()}
            
            if not self.seed_stats or not self.matchup_stats:
                self.logger.warning("Loaded cache is incomplete, initializing with default values")
                self._initialize_defaults()
                return False
                
            self.logger.info("Successfully loaded seed analysis from cache")
            return True
        except FileNotFoundError:
            self.logger.warning("Cache file not found, initializing with default values")
            self._initialize_defaults()
            return False
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            self._initialize_defaults()
            return False

    def _initialize_defaults(self):
        """Initialize default seed statistics based on historical trends."""
        # Set up basic seed stats based on general historical performance
        self.seed_stats = {i: {
            'wins': 0,
            'losses': 0,
            'games': 0,
            'round_wins': {
                'First Round': 0,
                'Second Round': 0,
                'Sweet Sixteen': 0,
                'Elite Eight': 0,
                'Final Four': 0,
                'Championship': 0
            },
            'championships': 0,
            'final_fours': 0,
            'elite_eights': 0,
            'sweet_sixteens': 0,
            'win_rate': 0.7 - ((i-1) * 0.04),  # Approximate win rate that decreases by seed
            'championship_rate': max(0, 0.2 - ((i-1) * 0.02)),  # Higher seeds have better championship odds
            'final_four_rate': max(0, 0.3 - ((i-1) * 0.025)),  # Higher seeds have better Final Four odds
            'elite_eight_rate': max(0, 0.4 - ((i-1) * 0.03)),  # Higher seeds have better Elite Eight odds
            'sweet_sixteen_rate': max(0, 0.5 - ((i-1) * 0.035))  # Higher seeds have better Sweet Sixteen odds
        } for i in range(1, 17)}  # Seeds 1-16

        # Initialize matchup stats
        self.matchup_stats = {}
        for i in range(1, 17):
            for j in range(1, 17):
                # Higher seeds generally have advantage over lower seeds
                win_prob = 0.5
                if i < j:  # i is higher seed (lower number)
                    win_prob = 0.5 + ((j - i) * 0.05)  # Advantage increases with seed difference
                elif i > j:  # i is lower seed (higher number)
                    win_prob = 0.5 - ((i - j) * 0.05)  # Disadvantage increases with seed difference
                
                # Some well-known upset patterns
                if i == 12 and j == 5:
                    win_prob = 0.35  # 12 seeds have historically done well against 5 seeds
                elif i == 11 and j == 6:
                    win_prob = 0.4  # 11 seeds often upset 6 seeds
                elif i == 10 and j == 7:
                    win_prob = 0.45  # 10-7 matchups are often close
                elif i == 9 and j == 8:
                    win_prob = 0.5  # 9-8 matchups are essentially toss-ups
                
                win_prob = min(0.95, max(0.05, win_prob))  # Cap probabilities
                
                self.matchup_stats[f"{i}-{j}"] = {
                    'wins': 0,
                    'total': 0,
                    'win_probability': win_prob
                }
                
        # Save the default values to cache
        self._save_cache()

    def _save_cache(self) -> None:
        """Save seed analysis data to cache."""
        data = {
            'seed_stats': self.seed_stats,
            'matchup_stats': self.matchup_stats,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info("Seed analysis data saved to cache")
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def get_seed_win_probability(self, seed1: int, seed2: int) -> float:
        """Get the historical probability of seed1 beating seed2."""
        matchup_key = f"{seed1}-{seed2}"
        stats = self.matchup_stats.get(matchup_key, {})
        
        if stats and 'win_probability' in stats:
            return stats['win_probability']
        
        # If no direct matchup data, use relative seed strength
        seed1_stats = self.seed_stats.get(seed1, {})
        seed2_stats = self.seed_stats.get(seed2, {})
        
        seed1_win_rate = seed1_stats.get('win_rate', 0)
        seed2_win_rate = seed2_stats.get('win_rate', 0)
        
        if seed1_win_rate + seed2_win_rate > 0:
            return seed1_win_rate / (seed1_win_rate + seed2_win_rate)
        
        # If no win rate data, use seed difference
        return 1 - (seed1 / (seed1 + seed2))

    def get_seed_round_probability(self, seed: int, round_name: str) -> float:
        """Get the historical probability of a seed reaching/winning in a specific round."""
        stats = self.seed_stats.get(seed, {})
        
        if round_name == "First Round":
            return stats.get('win_rate', 0.5)
        elif round_name == "Second Round":
            return stats.get('sweet_sixteen_rate', 0.3)
        elif round_name == "Sweet Sixteen":
            return stats.get('elite_eight_rate', 0.2)
        elif round_name == "Elite Eight":
            return stats.get('final_four_rate', 0.1)
        elif round_name == "Final Four":
            return stats.get('championship_rate', 0.05)
        elif round_name == "Championship Game":
            return max(0.01, stats.get('championship_rate', 0.05) * 0.5)
            
        return 0.1  # Default value for unknown rounds

    def get_seed_metrics(self, seed: int) -> Dict:
        """Get all metrics for a specific seed."""
        return self.seed_stats.get(seed, {})

    def get_upset_probability(self, higher_seed: int, lower_seed: int) -> float:
        """Calculate the probability of an upset (higher seed beating lower seed)."""
        if higher_seed <= lower_seed:
            return 0.0  # Not an upset
            
        matchup_key = f"{higher_seed}-{lower_seed}"
        stats = self.matchup_stats.get(matchup_key, {})
        
        if stats and 'win_probability' in stats:
            return stats['win_probability']
        
        # If no direct matchup data, calculate based on seed difference
        seed_diff = higher_seed - lower_seed
        base_upset_prob = 0.2  # Base upset probability
        
        # Well-known upset seeds get a boost
        if higher_seed == 12 and lower_seed == 5:
            return 0.35  # 12 over 5 is a common upset
        elif higher_seed == 11 and lower_seed == 6:
            return 0.32  # 11 over 6 is fairly common
        elif higher_seed == 10 and lower_seed == 7:
            return 0.4  # 10 over 7 happens often
        elif higher_seed == 9 and lower_seed == 8:
            return 0.5  # 9-8 is essentially a toss-up
            
        # For other matchups, probability decreases as seed difference increases
        return max(0.05, base_upset_prob - (seed_diff * 0.01))

def main():
    analyzer = SeedAnalyzer()
    analyzer.load_cache()
    
    # Print some example probabilities
    print("Seed matchup win probabilities:")
    print(f"1 seed vs 16 seed: {analyzer.get_seed_win_probability(1, 16):.1%}")
    print(f"8 seed vs 9 seed: {analyzer.get_seed_win_probability(8, 9):.1%}")
    print(f"5 seed vs 12 seed: {analyzer.get_seed_win_probability(5, 12):.1%}")
    
    # Print some upset probabilities
    print("\nUpset probabilities:")
    print(f"12 seed over 5 seed: {analyzer.get_upset_probability(12, 5):.1%}")
    print(f"11 seed over 6 seed: {analyzer.get_upset_probability(11, 6):.1%}")
    print(f"16 seed over 1 seed: {analyzer.get_upset_probability(16, 1):.1%}")
    
    # Print round probabilities
    print("\nRound advancement probabilities for a 1 seed:")
    seed = 1
    for round_name in ["First Round", "Second Round", "Sweet Sixteen", "Elite Eight", "Final Four", "Championship Game"]:
        prob = analyzer.get_seed_round_probability(seed, round_name)
        print(f"{round_name}: {prob:.1%}")

if __name__ == "__main__":
    main() 