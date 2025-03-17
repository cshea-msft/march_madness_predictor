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

    def analyze_tournament_data(self, tournament_data: List[Dict]) -> None:
        """Analyze tournament data to calculate seed-based statistics."""
        # Initialize seed stats
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
            'sweet_sixteens': 0
        } for i in range(1, 17)}  # Seeds 1-16

        # Initialize matchup stats
        self.matchup_stats = {}
        for i in range(1, 17):
            for j in range(1, 17):
                self.matchup_stats[f"{i}-{j}"] = {
                    'wins': 0,
                    'total': 0
                }

        # Process each game
        for game in tournament_data:
            winner_seed = game['winner_seed']
            loser_seed = game['loser_seed']
            round_name = game['round']

            if winner_seed == 0 or loser_seed == 0:
                continue  # Skip games with unknown seeds

            # Update seed stats
            self._update_seed_stats(winner_seed, loser_seed, round_name)
            
            # Update matchup stats
            self._update_matchup_stats(winner_seed, loser_seed)

        # Calculate advanced metrics
        self._calculate_advanced_metrics()
        
        # Save to cache
        self._save_cache()

    def _update_seed_stats(self, winner_seed: int, loser_seed: int, round_name: str) -> None:
        """Update statistics for individual seeds."""
        # Update winner stats
        self.seed_stats[winner_seed]['wins'] += 1
        self.seed_stats[winner_seed]['games'] += 1
        self.seed_stats[winner_seed]['round_wins'][round_name] += 1

        # Update specific achievement counts
        if round_name == 'Championship':
            self.seed_stats[winner_seed]['championships'] += 1
        elif round_name == 'Final Four':
            self.seed_stats[winner_seed]['final_fours'] += 1
        elif round_name == 'Elite Eight':
            self.seed_stats[winner_seed]['elite_eights'] += 1
        elif round_name == 'Sweet Sixteen':
            self.seed_stats[winner_seed]['sweet_sixteens'] += 1

        # Update loser stats
        self.seed_stats[loser_seed]['losses'] += 1
        self.seed_stats[loser_seed]['games'] += 1

    def _update_matchup_stats(self, winner_seed: int, loser_seed: int) -> None:
        """Update head-to-head matchup statistics."""
        matchup_key = f"{winner_seed}-{loser_seed}"
        reverse_key = f"{loser_seed}-{winner_seed}"

        self.matchup_stats[matchup_key]['wins'] += 1
        self.matchup_stats[matchup_key]['total'] += 1
        self.matchup_stats[reverse_key]['total'] += 1

    def _calculate_advanced_metrics(self) -> None:
        """Calculate advanced metrics for seeds and matchups."""
        # Calculate seed performance metrics
        for seed in self.seed_stats:
            stats = self.seed_stats[seed]
            games = stats['games']
            if games > 0:
                stats['win_rate'] = stats['wins'] / games
                stats['championship_rate'] = stats['championships'] / games
                stats['final_four_rate'] = stats['final_fours'] / games
                stats['elite_eight_rate'] = stats['elite_eights'] / games
                stats['sweet_sixteen_rate'] = stats['sweet_sixteens'] / games

        # Calculate matchup probabilities
        for matchup in self.matchup_stats:
            stats = self.matchup_stats[matchup]
            if stats['total'] > 0:
                stats['win_probability'] = stats['wins'] / stats['total']

    def get_seed_win_probability(self, seed1: int, seed2: int) -> float:
        """Get the historical probability of seed1 beating seed2."""
        matchup_key = f"{seed1}-{seed2}"
        stats = self.matchup_stats.get(matchup_key, {})
        
        if stats.get('total', 0) > 0:
            return stats['win_probability']
        
        # If no direct matchup data, use relative seed strength
        seed1_stats = self.seed_stats.get(seed1, {})
        seed2_stats = self.seed_stats.get(seed2, {})
        
        seed1_win_rate = seed1_stats.get('win_rate', 0)
        seed2_win_rate = seed2_stats.get('win_rate', 0)
        
        if seed1_win_rate + seed2_win_rate > 0:
            return seed1_win_rate / (seed1_win_rate + seed2_win_rate)
        
        # If no historical data, use seed difference
        return 1 - (seed1 / (seed1 + seed2))

    def get_seed_round_probability(self, seed: int, round_name: str) -> float:
        """Get the historical probability of a seed reaching/winning in a specific round."""
        stats = self.seed_stats.get(seed, {})
        round_wins = stats.get('round_wins', {}).get(round_name, 0)
        games = stats.get('games', 0)
        
        if games > 0:
            return round_wins / games
        return 0

    def get_seed_metrics(self, seed: int) -> Dict:
        """Get all metrics for a specific seed."""
        return self.seed_stats.get(seed, {})

    def get_upset_probability(self, higher_seed: int, lower_seed: int) -> float:
        """Calculate the probability of an upset (higher seed beating lower seed)."""
        if higher_seed <= lower_seed:
            return 0.0  # Not an upset
            
        matchup_key = f"{higher_seed}-{lower_seed}"
        stats = self.matchup_stats.get(matchup_key, {})
        
        if stats.get('total', 0) > 0:
            return stats['win_probability']
        
        # If no direct matchup data, use historical upset rates
        higher_stats = self.seed_stats.get(higher_seed, {})
        upset_games = sum(1 for m, s in self.matchup_stats.items() 
                         if int(m.split('-')[0]) == higher_seed and 
                         int(m.split('-')[1]) < higher_seed and 
                         s['wins'] > 0)
        total_games = higher_stats.get('games', 0)
        
        return upset_games / total_games if total_games > 0 else 0.1  # Default 10% upset chance

    def _save_cache(self) -> None:
        """Save analysis results to cache file."""
        try:
            cache_data = {
                'seed_stats': self.seed_stats,
                'matchup_stats': self.matchup_stats,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving seed analysis cache: {str(e)}")

    def load_cache(self) -> bool:
        """Load analysis results from cache file."""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                self.seed_stats = cache_data['seed_stats']
                self.matchup_stats = cache_data['matchup_stats']
                return True
        except FileNotFoundError:
            return False
        except Exception as e:
            self.logger.error(f"Error loading seed analysis cache: {str(e)}")
            return False 