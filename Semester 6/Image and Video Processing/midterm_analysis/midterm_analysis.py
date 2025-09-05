import pandas as pd
import statistics
import matplotlib.pyplot as plt

df = pd.read_csv("midterm_analysis/midterm_scores.tsv", delimiter='\t', index_col=0)
total_series = pd.Series(df['Total'].values, index=df['Roll no.'].astype(str))

# Filter scores based on roll numbers
scores_2021 = total_series[total_series.index.str.startswith('2021')].tolist()
scores_2022 = total_series[total_series.index.str.startswith('2022')].tolist()

# Calculate statistics for 2021 scores
mean_score_2021 = statistics.mean(scores_2021)
median_score_2021 = statistics.median(scores_2021)
mode_score_2021 = statistics.mode(scores_2021)
std_dev_score_2021 = statistics.stdev(scores_2021)
highest_score_2021 = max(scores_2021)
lowest_score_2021 = min(scores_2021)

# Calculate statistics for 2022 scores
mean_score_2022 = statistics.mean(scores_2022)
median_score_2022 = statistics.median(scores_2022)
mode_score_2022 = statistics.mode(scores_2022)
std_dev_score_2022 = statistics.stdev(scores_2022)
highest_score_2022 = max(scores_2022)
lowest_score_2022 = min(scores_2022)

# Print statistics side by side
print(f"{'Statistic':<20} {'2021 Scores':<20} {'2022 Scores':<20}")
print(f"{'-'*60}")
print(f"{'Mean':<20} {mean_score_2021:<20} {mean_score_2022:<20}")
print(f"{'Median':<20} {median_score_2021:<20} {median_score_2022:<20}")
print(f"{'Mode':<20} {mode_score_2021:<20} {mode_score_2022:<20}")
print(f"{'Standard Deviation':<20} {std_dev_score_2021:<20} {std_dev_score_2022:<20}")
print(f"{'Highest Score':<20} {highest_score_2021:<20} {highest_score_2022:<20}")
print(f"{'Lowest Score':<20} {lowest_score_2021:<20} {lowest_score_2022:<20}")

# Plotting the histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(scores_2021, bins=10, edgecolor='black')
plt.title('Histogram of 2021 Scores')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.xlim(0, 30)
plt.ylim(0,8)

plt.subplot(1, 2, 2)
plt.hist(scores_2022, bins=10, edgecolor='black')
plt.title('Histogram of 2022 Scores')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.xlim(0, 30)
plt.ylim(0,8)

plt.tight_layout()
plt.savefig("midterm_analysis/midterm_distribution_comparison.png", dpi=300)
#plt.show()
