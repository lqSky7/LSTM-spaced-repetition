1. difficulty (float)
Range: 0.0 to 2.0 (though the API description says 0-10, the actual data/model uses 0-2)
Description: The inherent difficulty level of the DSA problem.
0.0 = Easy (simple problems, e.g., basic array operations)
1.0 = Medium (moderate complexity, e.g., involving sorting or basic DP)
2.0 = Hard (complex problems, e.g., advanced graphs or optimization)
Usage in Model: A static feature per problem. Higher values indicate greater forgetting rates and longer initial review intervals. Influences the exponential decay parameter in the LSTM.
2. category (float)
Range: 0.0 to 14.0 (integer indices into a predefined category list)
Description: The primary DSA category/topic of the problem, encoded as an index.
Examples: 0 = Arrays, 1 = Strings, 2 = LinkedList, 3 = Trees, 4 = Graphs, 5 = DP, 6 = Greedy, 7 = Backtracking, 8 = Sorting, 9 = Searching, 10 = Stack, 11 = Queue, 12 = Heap, 13 = HashMap, 14 = Math.
Usage in Model: Categorical feature to capture topic-specific learning patterns. Helps the model differentiate between areas like "Graphs" (often harder retention) vs. "Arrays" (easier).
3. concept_count (float)
Range: 1.0 to 6.0 (typically 1-7 in data, but capped in practice)
Description: The number of distinct DSA concepts required to solve the problem (e.g., a problem needing "arrays + sorting + recursion" has concept_count = 3.0).
Usage in Model: Indicates problem complexity. Higher counts correlate with increased difficulty and shorter review intervals, as multi-concept problems are harder to master and retain.
4. attempt_number (float)
Range: 1.0 and up (sequential per user-problem pair)
Description: The ordinal number of this attempt for the specific user-problem combination (e.g., 1.0 for the first try, 2.0 for the second review).
Usage in Model: Tracks progression. Early attempts (low numbers) have different patterns than later reviews, influencing spacing effects and cumulative success.
5. days_since_last_attempt (float)
Range: 0.0 and up (days, can be fractional but usually integer)
Description: The number of days elapsed since the user's previous attempt on this problem (0.0 for the first attempt).
Usage in Model: Critical for spaced repetition. Used in the exponential decay formula to model forgetting curves (Ebbinghaus-inspired). Longer gaps increase predicted intervals if mastery is high.
6. num_tries (float)
Range: 1.0 and up (typically 1-15)
Description: The number of submission attempts (e.g., "Run" or "Submit" clicks) made during this session before success or giving up.
Usage in Model: Reflects struggle within an attempt. Higher values indicate more retries, suggesting weaker recall and potentially shorter next intervals.
7. time_spent_minutes (float)
Range: 5.0 to 180.0 (minutes, clamped in data)
Description: Total time spent on this attempt, including thinking, coding, and testing.
Usage in Model: Proxy for effort and engagement. Longer times may indicate difficulty or thorough learning, affecting interval predictions.
8. num_mistakes (float)
Range: 0.0 and up (count of errors, derived from mistake tags)
Description: The number of distinct mistakes/errors made during the attempt (e.g., 5.0 for issues like "off-by-one, logic-error, boundary-conditions, etc.").
Usage in Model: Measures error frequency. Higher counts signal poor performance, leading to shorter review intervals to reinforce weak areas.
9. cumulative_success_rate (float)
Range: 0.0 to 1.0
Description: The running success rate for this user-problem pair up to this attempt (e.g., 0.0 if all previous attempts failed, 0.5 for 1/2 successes).
Usage in Model: Tracks overall mastery. Higher rates suggest better retention, allowing longer intervals. Calculated as total_successes / total_attempts for the pair.
Additional Notes
Sequence Context: The API takes a list of these records (the attempt history) to form an LSTM sequence. The model predicts the interval after the last attempt.
Normalization: All values are scaled using a StandardScaler (fitted on training data) before feeding into the LSTM, so raw values are transformed internally.
Model Output: Predicts next_review_interval_days (1-90 days) using exponential decay to target ~90% recall probability.
Validation: The API uses Pydantic for type checking. Ensure floats are used (e.g., 2.0 not 2).
Source: These fields match the feature_cols in the training script, derived from the CSV datasets.
If you need examples, endpoint details, or modifications, let me know!

Grok Code Fast 1 • 0x
