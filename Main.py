import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import auc, roc_curve


# --- Get data from the CSV file ---
original_data = pd.read_csv("diabetes_risk.csv")

# Separate data in train, validation and test sets
separation_values = [0.7, 0.15, 0.15]  # Train, Validation, Test
data = original_data.sample(frac=separation_values[0], random_state=42)
validation_data = original_data.sample(frac=separation_values[1], random_state=42, replace=False)
test_data = original_data.sample(frac=separation_values[2], random_state=42, replace=False)

# Seperate by classes
class_column = data.columns[-1]
classes = sorted(data[class_column].unique())

# Select numerical and categorical features
numerical_columns = data.select_dtypes(include='number').columns
numerical_features = numerical_columns[~numerical_columns.isin([data.columns[0], class_column])]  # Exclude the ID and class columns
categorical_columns = data.select_dtypes(include='object').columns
categorical_features = categorical_columns[~categorical_columns.isin([data.columns[0], class_column])]  # Exclude the ID and class columns


# --- Feature Selection and Reduction ---
# Kruskall-Wallis test for numerical features
def kruskal_wallis_test(feature):
    # Prepare the data for each class
    values_by_class = [
        data.loc[data[class_column] == class_value, feature].dropna().values
        for class_value in classes
    ]
    
    # Calculate ranks for all values combined
    all_values = pd.concat([pd.Series(v) for v in values_by_class], ignore_index=True)
    ranks = all_values.rank().values  # average ranks for ties
    n_values = [len(values) for values in values_by_class] # number of observations for each class

    # Calculate the sum of ranks for each class
    rank_sums = []
    start = 0
    for values in values_by_class:
        end = start + len(values)
        rank_sums.append(ranks[start:end].sum())
        start = end
    
    # Calculate the Kruskal-Wallis H statistic
    N = sum(n_values)
    rank_means = [rank_sum / n for rank_sum, n in zip(rank_sums, n_values)]
    
    overall_rank_mean = (N + 1) / 2
    H = (12 / (N * (N + 1))) * sum(n * (rank_mean - overall_rank_mean) ** 2 for n, rank_mean in zip(n_values, rank_means))
    return H

h_statistics = {feature: kruskal_wallis_test(feature) for feature in numerical_features}
sorted_features = sorted(h_statistics.items(), key=lambda x: x[1], reverse=True)

'''print("Features ordered by Kruskal-Wallis H statistic:")
for feature, h in sorted_features:
    print(f"{feature}: H = {h:.4f}")'''

# Correlation matrix for the top features
top_features = [feature for feature, _ in sorted_features[:5]]  # Select top 5 features based on H statistic
correlation_matrix = data[top_features].corr()

# Plot the Correlation Matrix 
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.yticks(range(len(top_features)), top_features)

for row in range(correlation_matrix.shape[0]):
    for column in range(correlation_matrix.shape[1]):
        plt.text(
            column,
            row,
            f'{correlation_matrix.iloc[row, column]:.4f}',
            ha='center',
            va='center',
            color='black',
        )

plt.title('Correlation Matrix')
plt.show()

# ROC Curve and AUC
roc_targets = (data[class_column] == "High").astype(int).to_numpy()

roc_results = {}
plt.figure(figsize=(10, 7))

for feature in numerical_features:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        roc_targets,
        data[feature].to_numpy(),
    )
    feature_auc = auc(false_positive_rate, true_positive_rate)
    roc_results[feature] = feature_auc
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f'{feature} (AUC = {feature_auc:.4f})',
    )

plt.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.5000)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves: Super versus Average')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

sorted_auc = sorted(roc_results.items(), key=lambda item: item[1], reverse=True)
'''print("Features ordered by ROC AUC:")
for feature, feature_auc in sorted_auc:
    print(f"{feature}: AUC = {feature_auc:.4f}")'''

# Compare Kruskal-Wallis H and ROC AUC
print("\nComparison of Kruskal-Wallis H and ROC AUC:")
comparison = pd.DataFrame({
    'Kruskal-Wallis H': h_statistics,
    'ROC AUC': roc_results,
}).sort_values('Kruskal-Wallis H', ascending=False)
print(comparison)