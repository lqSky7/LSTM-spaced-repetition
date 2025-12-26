"""
Synthetic DSA Spaced Repetition Dataset Generator
==================================================
Generates realistic learning sequences based on cognitive science principles
including Ebbinghaus forgetting curve, spacing effect, and performance-based scheduling.

Usage:
    python generate_dsa_dataset.py

Output:
    dsa_synthetic_dataset.csv - Complete dataset with all features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple


class DSADatasetGenerator:
    """Generate synthetic DSA spaced repetition learning data"""

    # Problem configuration
    CATEGORIES = [
        'Arrays', 'Strings', 'LinkedList', 'Trees', 'Graphs', 
        'DP', 'Greedy', 'Backtracking', 'Sorting', 'Searching',
        'Stack', 'Queue', 'Heap', 'HashMap', 'Math'
    ]

    MISTAKE_TAGS = [
        'off-by-one', 'edge-cases', 'null-pointer', 'optimization',
        'time-complexity', 'space-complexity', 'logic-error',
        'boundary-conditions', 'algorithm-choice', 'implementation',
        'recursion-base-case', 'loop-termination', 'overflow',
        'memory-management', 'input-validation'
    ]

    DIFFICULTY_LEVELS = ['Easy', 'Medium', 'Hard']

    def __init__(self, seed=None):
        """Initialize generator with optional random seed"""
        if seed is not None:
            np.random.seed(seed)

        self.problems_db = {}  # Store problem metadata

    def _initialize_problem(self, problem_id: int) -> Dict:
        """Create a new problem with random characteristics"""
        difficulty = np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
        category = np.random.randint(0, len(self.CATEGORIES))

        # Concept count: harder problems require more concepts
        concept_count = np.random.randint(1, 4) if difficulty == 0 else                        np.random.randint(2, 5) if difficulty == 1 else                        np.random.randint(3, 7)

        # Estimated time complexity and solution length
        time_complexity_class = np.random.choice([1, 2, 3, 4, 5])  # O(n) to O(n^2) etc
        code_lines = np.random.randint(10, 30) if difficulty == 0 else                      np.random.randint(20, 60) if difficulty == 1 else                      np.random.randint(40, 120)

        return {
            'difficulty': difficulty,
            'category': category,
            'concept_count': concept_count,
            'time_complexity_class': time_complexity_class,
            'code_lines': code_lines,
            'attempts_history': [],
            'last_attempt_date': None,
            'total_correct': 0,
            'total_attempts': 0
        }

    def _calculate_initial_mastery(self, difficulty: int, learner_ability: float) -> float:
        """Calculate initial mastery probability for first attempt"""
        # Learner ability ranges from 0.3 (beginner) to 0.9 (expert)
        base_mastery = {0: 0.7, 1: 0.5, 2: 0.3}[difficulty]
        mastery = base_mastery * learner_ability + np.random.normal(0, 0.1)
        return np.clip(mastery, 0.05, 0.95)

    def _apply_forgetting_curve(
        self, 
        initial_strength: float, 
        days_elapsed: int,
        difficulty: int,
        successful_reviews: int
    ) -> float:
        """
        Apply Ebbinghaus forgetting curve with spacing effect
        R(t) = S * (1 + H*t)^-c where:
        - R(t) is recall probability at time t
        - S is initial strength
        - H is difficulty-based decay rate
        - c is decay exponent (typically ~0.5)
        """
        # Base decay rates (higher = faster forgetting)
        decay_rates = {0: 0.15, 1: 0.25, 2: 0.40}
        H = decay_rates[difficulty]

        # Spacing effect: each successful review reduces decay rate
        spacing_bonus = min(0.12, successful_reviews * 0.03)
        adjusted_H = H * (1 - spacing_bonus)

        # Power law decay
        c = 0.5
        recall_prob = initial_strength * (1 + adjusted_H * days_elapsed) ** (-c)

        return np.clip(recall_prob, 0.05, 0.95)

    def _calculate_performance_score(
        self,
        recall_prob: float,
        difficulty: int,
        learner_consistency: float
    ) -> Tuple[int, int, int, List[str]]:
        """
        Determine attempt outcome, tries, time, and mistakes based on recall probability
        Returns: (outcome, num_tries, time_spent, mistake_tags)
        """
        # Add noise based on learner consistency (0.5 = inconsistent, 1.0 = very consistent)
        noisy_recall = recall_prob + np.random.normal(0, 0.15 * (1 - learner_consistency))
        outcome = 1 if noisy_recall > 0.5 else 0

        # Number of tries
        if outcome == 1:
            # Successful: fewer tries, influenced by recall strength
            mean_tries = 1.5 + (1 - recall_prob) * 2 + difficulty * 0.5
            num_tries = max(1, int(np.random.exponential(mean_tries)))
        else:
            # Failed: more tries, especially for medium difficulty (not giving up)
            mean_tries = 3.0 + difficulty * 1.5 + (1 - recall_prob) * 2
            num_tries = max(1, int(np.random.exponential(mean_tries)))
        num_tries = min(num_tries, 15)

        # Time spent (minutes) - scales with difficulty and tries
        base_time = {0: 20, 1: 35, 2: 60}[difficulty]
        time_variance = base_time * 0.4
        time_spent = int(np.random.normal(
            base_time * (num_tries ** 0.6), 
            time_variance
        ))
        time_spent = max(5, min(time_spent, 180))  # 5 min to 3 hours

        # Mistake tags
        if outcome == 0:
            # Failed attempts have more mistakes
            num_mistakes = np.random.randint(1, min(4 + difficulty, len(self.MISTAKE_TAGS)))
            mistake_tags = list(np.random.choice(
                self.MISTAKE_TAGS, 
                size=num_mistakes, 
                replace=False
            ))
        else:
            # Successful attempts may still have minor mistakes
            if recall_prob < 0.7 or difficulty == 2:
                num_mistakes = np.random.randint(0, 3)
                mistake_tags = list(np.random.choice(
                    self.MISTAKE_TAGS, 
                    size=num_mistakes, 
                    replace=False
                )) if num_mistakes > 0 else []
            else:
                mistake_tags = []

        return outcome, num_tries, time_spent, mistake_tags

    def _calculate_next_review_interval(
        self,
        outcome: int,
        difficulty: int,
        attempt_number: int,
        current_success_rate: float,
        days_since_last: int
    ) -> int:
        """
        Calculate optimal days until next review using SM-2 inspired algorithm
        """
        # Base intervals by difficulty
        base_intervals = {0: 7, 1: 5, 2: 3}
        base = base_intervals[difficulty]

        if outcome == 1:
            # Successful review - exponential backoff with quality factor
            quality_factor = 1.3 + current_success_rate * 0.5
            interval = base * (quality_factor ** (attempt_number - 1))

            # Add slight randomness for realistic variance
            interval *= np.random.uniform(0.85, 1.15)
        else:
            # Failed review - short interval for relearning
            interval = max(1, base * 0.2 * np.random.uniform(0.5, 1.2))

        return max(1, min(int(interval), 90))  # Cap at 90 days

    def _select_problem_for_review(
        self, 
        problem_history: Dict[int, Dict],
        current_date: datetime,
        new_problem_probability: float
    ) -> int:
        """Select which problem to attempt next (new or review)"""
        if not problem_history or np.random.random() < new_problem_probability:
            # New problem
            return np.random.randint(0, 1000)  # 1000 possible problems

        # Review existing problem - weighted by time since last attempt
        problem_ids = list(problem_history.keys())
        weights = []

        for pid in problem_ids:
            last_date = problem_history[pid]['last_attempt_date']
            days_since = (current_date - last_date).days if last_date else 999
            # Higher weight for problems not seen recently
            weight = days_since ** 1.5
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        return np.random.choice(problem_ids, p=weights)

    def generate_user_sequence(
        self,
        user_id: int,
        learner_profile: str,
        num_attempts: int,
        start_date: datetime
    ) -> List[Dict]:
        """Generate complete learning sequence for one user"""

        # Learner profile parameters
        profiles = {
            'beginner': {'ability': 0.4, 'consistency': 0.6, 'new_prob': 0.4},
            'intermediate': {'ability': 0.65, 'consistency': 0.75, 'new_prob': 0.3},
            'advanced': {'ability': 0.85, 'consistency': 0.85, 'new_prob': 0.25},
            'inconsistent': {'ability': 0.5, 'consistency': 0.4, 'new_prob': 0.35},
            'fast_learner': {'ability': 0.75, 'consistency': 0.9, 'new_prob': 0.45}
        }

        profile_params = profiles.get(learner_profile, profiles['intermediate'])
        learner_ability = profile_params['ability'] + np.random.normal(0, 0.1)
        learner_ability = np.clip(learner_ability, 0.2, 0.95)
        learner_consistency = profile_params['consistency']
        new_problem_prob = profile_params['new_prob']

        problem_history = {}
        sequences = []
        current_date = start_date

        for attempt_idx in range(num_attempts):
            # Select problem
            problem_id = self._select_problem_for_review(
                problem_history, current_date, new_problem_prob
            )

            # Initialize problem if new
            if problem_id not in problem_history:
                problem_history[problem_id] = self._initialize_problem(problem_id)

            problem = problem_history[problem_id]
            difficulty = problem['difficulty']

            # Calculate days since last attempt
            if problem['last_attempt_date']:
                days_since_last = (current_date - problem['last_attempt_date']).days
            else:
                days_since_last = 0

            attempt_number = problem['total_attempts'] + 1

            # Calculate recall probability
            if attempt_number == 1:
                # First attempt
                recall_prob = self._calculate_initial_mastery(difficulty, learner_ability)
            else:
                # Subsequent attempts - apply forgetting curve
                initial_strength = problem['total_correct'] / problem['total_attempts']
                successful_reviews = problem['total_correct']
                recall_prob = self._apply_forgetting_curve(
                    initial_strength, days_since_last, difficulty, successful_reviews
                )

            # Generate performance metrics
            outcome, num_tries, time_spent, mistake_tags = self._calculate_performance_score(
                recall_prob, difficulty, learner_consistency
            )

            # Update problem history
            problem['total_attempts'] += 1
            problem['total_correct'] += outcome
            problem['attempts_history'].append(outcome)
            problem['last_attempt_date'] = current_date

            current_success_rate = problem['total_correct'] / problem['total_attempts']

            # Calculate next review interval
            review_interval = self._calculate_next_review_interval(
                outcome, difficulty, attempt_number, current_success_rate, days_since_last
            )

            # Additional derived features
            streak = 0
            for i in range(len(problem['attempts_history']) - 1, -1, -1):
                if problem['attempts_history'][i] == 1:
                    streak += 1
                else:
                    break

            # Record data point
            sequences.append({
                'user_id': user_id,
                'problem_id': problem_id,
                'difficulty': difficulty,
                'difficulty_label': self.DIFFICULTY_LEVELS[difficulty],
                'category': problem['category'],
                'category_label': self.CATEGORIES[problem['category']],
                'concept_count': problem['concept_count'],
                'time_complexity_class': problem['time_complexity_class'],
                'code_lines': problem['code_lines'],
                'attempt_number': attempt_number,
                'days_since_last_attempt': days_since_last,
                'outcome': outcome,
                'num_tries': num_tries,
                'time_spent_minutes': time_spent,
                'mistake_tags': ','.join(mistake_tags),
                'num_mistakes': len(mistake_tags),
                'cumulative_success_rate': current_success_rate,
                'success_streak': streak,
                'review_interval': review_interval,
                'recall_probability': round(recall_prob, 4),
                'timestamp': current_date.strftime('%Y-%m-%d'),
                'learner_profile': learner_profile
            })

            # Advance time: typically 1-5 days between attempts, more variation for inconsistent learners
            if learner_consistency > 0.7:
                days_forward = int(np.random.randint(1, 6))
            else:
                days_forward = int(np.random.choice([1, 2, 3, 5, 7, 14], p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05]))

            current_date += timedelta(days=days_forward)

        return sequences

    def generate_dataset(
        self,
        num_users: int = 200,
        attempts_per_user: Tuple[int, int] = (80, 180),
        start_date: str = '2024-01-01',
        profile_distribution: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Generate complete dataset with multiple users

        Args:
            num_users: Number of users to simulate
            attempts_per_user: (min, max) attempts per user
            start_date: Starting date for simulations
            profile_distribution: Distribution of learner profiles

        Returns:
            DataFrame with all learning sequences
        """
        if profile_distribution is None:
            profile_distribution = {
                'beginner': 0.25,
                'intermediate': 0.35,
                'advanced': 0.20,
                'inconsistent': 0.10,
                'fast_learner': 0.10
            }

        profiles = list(profile_distribution.keys())
        probabilities = list(profile_distribution.values())

        all_sequences = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        print(f"Generating dataset for {num_users} users...")

        for user_id in range(1, num_users + 1):
            if user_id % 20 == 0:
                print(f"  Generated {user_id}/{num_users} users...")

            # Select learner profile
            profile = np.random.choice(profiles, p=probabilities)

            # Random number of attempts
            num_attempts = np.random.randint(attempts_per_user[0], attempts_per_user[1])

            # Stagger start dates slightly (within 30 days)
            user_start = start_dt + timedelta(days=int(np.random.randint(0, 30)))

            # Generate sequences
            user_sequences = self.generate_user_sequence(
                user_id, profile, num_attempts, user_start
            )
            all_sequences.extend(user_sequences)

        print(f"✓ Generated {len(all_sequences)} total learning records")

        return pd.DataFrame(all_sequences)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic DSA spaced repetition dataset'
    )
    parser.add_argument('--users', type=int, default=200,
                       help='Number of users to simulate (default: 200)')
    parser.add_argument('--min-attempts', type=int, default=80,
                       help='Minimum attempts per user (default: 80)')
    parser.add_argument('--max-attempts', type=int, default=180,
                       help='Maximum attempts per user (default: 180)')
    parser.add_argument('--output', type=str, default='dsa_synthetic_dataset.csv',
                       help='Output filename (default: dsa_synthetic_dataset.csv)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for reproducibility (default: 123)')

    args = parser.parse_args()

    # Generate dataset
    generator = DSADatasetGenerator(seed=args.seed)
    df = generator.generate_dataset(
        num_users=args.users,
        attempts_per_user=(args.min_attempts, args.max_attempts)
    )

    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total records: {len(df)}")
    print(f"Total users: {df['user_id'].nunique()}")
    print(f"Total problems: {df['problem_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nDifficulty distribution:")
    print(df['difficulty_label'].value_counts())
    print(f"\nLearner profile distribution:")
    print(df['learner_profile'].value_counts())
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())
    print(f"\nAverage metrics:")
    print(f"  - Attempts per problem: {df.groupby('problem_id')['attempt_number'].max().mean():.2f}")
    print(f"  - Time spent: {df['time_spent_minutes'].mean():.1f} minutes")
    print(f"  - Review interval: {df['review_interval'].mean():.1f} days")
    print(f"  - Success rate: {df['outcome'].mean():.2%}")

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n✓ Dataset saved to: {args.output}")

    # Display sample
    print(f"\nSample records:")
    print(df.head(10).to_string())


if __name__ == '__main__':
    main()