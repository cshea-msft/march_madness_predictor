import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime
import time
from typing import Dict, Any, List
import logging

class TournamentHistoryScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.history_cache_file = "tournament_history_cache.json"
        self.base_url = "https://www.sports-reference.com/cbb/postseason"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _fetch_page(self, url: str) -> str:
        """Fetch webpage content with error handling and rate limiting."""
        try:
            time.sleep(3)  # Rate limiting
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return ""

    def _parse_tournament_year(self, year: int) -> List[Dict[str, Any]]:
        """Parse tournament data for a specific year."""
        url = f"{self.base_url}/{year}-ncaa.html"
        html = self._fetch_page(url)
        if not html:
            return []

        tournament_data = []
        soup = BeautifulSoup(html, 'lxml')
        
        try:
            # Find tournament brackets
            brackets = soup.find_all('div', class_='bracket')
            
            if not brackets:
                # Try alternative structure
                games = soup.find_all('div', class_='game')
                if not games:
                    games = soup.select('table.teams tr')
                
                for game in games:
                    try:
                        if 'class' in game.attrs and 'game' in game['class']:
                            teams = game.find_all('div', class_='team')
                        else:
                            teams = game.find_all('td')
                            
                        if len(teams) >= 2:
                            winner = teams[0]
                            loser = teams[1]
                            
                            game_data = {
                                'year': year,
                                'round': self._determine_round_from_context(game),
                                'winner': winner.text.strip(),
                                'loser': loser.text.strip(),
                                'winner_seed': self._extract_seed(winner),
                                'loser_seed': self._extract_seed(loser)
                            }
                            tournament_data.append(game_data)
                    except Exception as e:
                        self.logger.error(f"Error parsing game in {year}: {str(e)}")
                        continue
            else:
                for bracket in brackets:
                    games = bracket.find_all('div', class_='game')
                    for game in games:
                        teams = game.find_all('div', class_='team')
                        if len(teams) == 2:
                            winner = teams[0].find('span', class_='winner')
                            loser = teams[1]
                            
                            game_data = {
                                'year': year,
                                'round': self._determine_round(bracket.get('class', [])),
                                'winner': winner.text.strip() if winner else "",
                                'loser': loser.text.strip(),
                                'winner_seed': self._extract_seed(teams[0]),
                                'loser_seed': self._extract_seed(teams[1])
                            }
                            tournament_data.append(game_data)
            
            return tournament_data
        except Exception as e:
            self.logger.error(f"Error parsing tournament data for {year}: {str(e)}")
            return []

    def _determine_round_from_context(self, game_element) -> str:
        """Determine tournament round from game context."""
        try:
            # Try to find round information in parent elements or attributes
            parent_text = ' '.join([p.text for p in game_element.parents if hasattr(p, 'text')])
            
            rounds = {
                'first': 'First Round',
                'second': 'Second Round',
                'sweet sixteen': 'Sweet Sixteen',
                'elite eight': 'Elite Eight',
                'final four': 'Final Four',
                'championship': 'Championship'
            }
            
            for key, value in rounds.items():
                if key in parent_text.lower():
                    return value
            
            return "Unknown Round"
        except:
            return "Unknown Round"

    def _determine_round(self, classes: List[str]) -> str:
        """Determine tournament round from bracket classes."""
        round_mapping = {
            'round1': 'First Round',
            'round2': 'Second Round',
            'round3': 'Sweet Sixteen',
            'round4': 'Elite Eight',
            'round5': 'Final Four',
            'round6': 'Championship'
        }
        
        for class_name in classes:
            if class_name in round_mapping:
                return round_mapping[class_name]
        return "Unknown Round"

    def _extract_seed(self, team_div) -> int:
        """Extract team seed from team div."""
        try:
            seed_span = team_div.find('span', class_='seed')
            return int(seed_span.text.strip()) if seed_span else 0
        except:
            return 0

    def get_historical_data(self, start_year: int = 2010) -> Dict[str, Any]:
        """
        Get historical tournament data from start_year to present.
        Tries to load from cache first, then scrapes if necessary.
        """
        cached_data = self._load_cache()
        current_year = datetime.now().year
        
        if cached_data and self._is_cache_valid(cached_data):
            return cached_data
        
        historical_data = {
            'tournaments': [],
            'team_stats': {},
            'last_updated': datetime.now().isoformat()
        }
        
        for year in range(start_year, current_year):
            self.logger.info(f"Scraping tournament data for {year}...")
            tournament_data = self._parse_tournament_year(year)
            historical_data['tournaments'].extend(tournament_data)
        
        # Calculate team statistics
        historical_data['team_stats'] = self._calculate_team_stats(historical_data['tournaments'])
        
        # Save to cache
        self._save_cache(historical_data)
        
        return historical_data

    def _calculate_team_stats(self, tournaments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate historical statistics for each team."""
        team_stats = {}
        
        for game in tournaments:
            # Update winner stats
            if game['winner'] not in team_stats:
                team_stats[game['winner']] = {
                    'appearances': 0,
                    'wins': 0,
                    'losses': 0,
                    'championships': 0,
                    'final_fours': 0,
                    'elite_eights': 0,
                    'sweet_sixteens': 0,
                    'upset_wins': 0,  # Winning as lower seed
                    'upset_losses': 0  # Losing as higher seed
                }
            
            if game['loser'] not in team_stats:
                team_stats[game['loser']] = {
                    'appearances': 0,
                    'wins': 0,
                    'losses': 0,
                    'championships': 0,
                    'final_fours': 0,
                    'elite_eights': 0,
                    'sweet_sixteens': 0,
                    'upset_wins': 0,
                    'upset_losses': 0
                }
            
            # Update basic stats
            team_stats[game['winner']]['wins'] += 1
            team_stats[game['loser']]['losses'] += 1
            
            # Update round-specific achievements
            if game['round'] == 'Championship':
                team_stats[game['winner']]['championships'] += 1
            elif game['round'] == 'Final Four':
                team_stats[game['winner']]['final_fours'] += 1
            elif game['round'] == 'Elite Eight':
                team_stats[game['winner']]['elite_eights'] += 1
            elif game['round'] == 'Sweet Sixteen':
                team_stats[game['winner']]['sweet_sixteens'] += 1
            
            # Update upset stats
            if game['winner_seed'] > game['loser_seed']:
                team_stats[game['winner']]['upset_wins'] += 1
                team_stats[game['loser']]['upset_losses'] += 1
        
        # Calculate appearance counts and success rates
        for team in team_stats:
            stats = team_stats[team]
            stats['appearances'] = stats['wins'] + stats['losses']
            stats['win_rate'] = stats['wins'] / stats['appearances'] if stats['appearances'] > 0 else 0
            stats['championship_rate'] = stats['championships'] / stats['appearances'] if stats['appearances'] > 0 else 0
            stats['final_four_rate'] = stats['final_fours'] / stats['appearances'] if stats['appearances'] > 0 else 0
            stats['upset_rate'] = stats['upset_wins'] / stats['wins'] if stats['wins'] > 0 else 0
        
        return team_stats

    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid (less than 30 days old)."""
        try:
            last_updated = datetime.fromisoformat(cached_data['last_updated'])
            age = datetime.now() - last_updated
            return age.days < 30
        except:
            return False

    def _save_cache(self, data: Dict[str, Any]) -> None:
        """Save data to cache file."""
        try:
            with open(self.history_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def _load_cache(self) -> Dict[str, Any]:
        """Load data from cache file."""
        try:
            with open(self.history_cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            return {} 