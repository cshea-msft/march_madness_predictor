#!/usr/bin/env python
from seed_analysis import SeedAnalyzer

def main():
    print("Initializing seed analyzer and generating cache...")
    analyzer = SeedAnalyzer()
    analyzer._initialize_defaults()
    
    # Print some example probabilities to verify
    print("\nSeed matchup win probabilities:")
    print(f"1 seed vs 16 seed: {analyzer.get_seed_win_probability(1, 16):.1%}")
    print(f"8 seed vs 9 seed: {analyzer.get_seed_win_probability(8, 9):.1%}")
    print(f"5 seed vs 12 seed: {analyzer.get_seed_win_probability(5, 12):.1%}")
    
    # Print some upset probabilities
    print("\nUpset probabilities:")
    print(f"12 seed over 5 seed: {analyzer.get_upset_probability(12, 5):.1%}")
    print(f"11 seed over 6 seed: {analyzer.get_upset_probability(11, 6):.1%}")
    print(f"16 seed over 1 seed: {analyzer.get_upset_probability(16, 1):.1%}")
    
    # Print championship rates
    print("\nChampionship rates:")
    for seed in range(1, 17):
        metrics = analyzer.get_seed_metrics(seed)
        print(f"Seed #{seed}: {metrics.get('championship_rate', 0):.1%}")
    
    print("\nCache file successfully generated at: seed_analysis_cache.json")

if __name__ == "__main__":
    main() 