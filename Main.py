import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import auc, roc_curve,  accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# --- Get data from the CSV file ---
original_data = pd.read_csv("diabetes_risk.csv")

# Separate data in train, validation and test sets
separation_values = [0.7, 0.15, 0.15]  # Train, Validation, Test
class_column = original_data.columns[-1]

data, remaining_data = train_test_split(
    original_data,
    train_size=separation_values[0],
    stratify=original_data[class_column],
    random_state=42,
)
validation_data, test_data = train_test_split(
    remaining_data,
    test_size=separation_values[2] / (separation_values[1] + separation_values[2]),
    stratify=remaining_data[class_column],
    random_state=42,
)

# Seperate by classes
classes = sorted(data[class_column].unique())

# Select numerical and categorical features
numerical_columns = data.select_dtypes(include='number').columns
numerical_features = numerical_columns[~numerical_columns.isin([data.columns[0], class_column])]  # Exclude the ID and class columns
categorical_columns = data.select_dtypes(include=['object', 'string', 'category']).columns
categorical_features = categorical_columns[~categorical_columns.isin([data.columns[0], class_column])]  # Exclude the ID and class columns


# --- Missing Values and Categorical Encoding ---
# Categorical features should be ranked too
raw_features = (list(numerical_features) + list(categorical_features))

X_train = data[raw_features].copy()
X_validation = validation_data[raw_features].copy()
X_test = test_data[raw_features].copy()

y_train = data[class_column]
y_validation = validation_data[class_column]
y_test = test_data[class_column]

# Represent missing categorical values as a separate category.
# The categories seen in training are used for every set, so that
# drop_first removes the same dummy column everywhere
for feature in categorical_features:
    train_categories = sorted(
        X_train[feature].astype('string').fillna('Missing').unique()
    )
    for dataset in [X_train, X_validation, X_test]:
        dataset[feature] = pd.Categorical(
            dataset[feature].astype('string').fillna('Missing'),
            categories=train_categories,
        )

# Convert categorical features into numerical columns
# (drop_first avoids perfectly collinear dummy columns)
X_train = pd.get_dummies(X_train, columns=list(categorical_features), dtype=float, drop_first=True)
X_validation = pd.get_dummies(X_validation, columns=list(categorical_features), dtype=float, drop_first=True)
X_test = pd.get_dummies(X_test, columns=list(categorical_features), dtype=float, drop_first=True)

# Keep the same columns and column order as the training set
X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0,)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0,)


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

    # Tie correction (needed for discrete and binary features)
    tie_counts = all_values.value_counts().to_numpy().astype(float)
    tie_correction = 1 - (tie_counts ** 3 - tie_counts).sum() / (N ** 3 - N)
    return H / tie_correction if tie_correction > 0 else 0.0   

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
plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
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
plt.tight_layout()
plt.show()

# Highly correlated pairs among all features (redundancy check)
full_correlation = X_train.corr().abs()
for i, first in enumerate(full_correlation.columns):
    for second in full_correlation.columns[i + 1:]:
        if full_correlation.loc[first, second] > 0.8:
            print(f"High correlation: {first} / {second} = {full_correlation.loc[first, second]:.4f}")

# ROC Curve and AUC
roc_targets = (data[class_column] == "High").astype(int).to_numpy()

roc_results = {}
roc_curves = {}
for feature in X_train.columns:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        roc_targets, X_train[feature].to_numpy()
    )
    roc_results[feature] = auc(false_positive_rate, true_positive_rate)
    roc_curves[feature] = (false_positive_rate, true_positive_rate)

plt.figure(figsize=(10, 7))
top_roc = sorted(
    roc_results,
    key=lambda f: max(roc_results[f], 1 - roc_results[f]), # Use the strength of the AUC, not the direction 
    reverse=True,
)[:10]  # Display the top 10 features

for feature in top_roc:
    plt.plot(
        *roc_curves[feature],
        linewidth=2,
        label=f'{feature} (AUC = {roc_results[feature]:.4f})',
    )

plt.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.5000)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves: High versus the other classes')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

sorted_auc = sorted(roc_results.items(), key=lambda item: item[1], reverse=True)
'''print("Features ordered by ROC AUC:")
for feature, feature_auc in sorted_auc:
    print(f"{feature}: AUC = {feature_auc:.4f}")'''

# Compare Kruskal-Wallis H and ROC AUC
# (AUC strength = max(AUC, 1 - AUC), since AUC < 0.5 means a negative relation)
print("\nComparison of Kruskal-Wallis H and ROC AUC:")
comparison = pd.DataFrame({
    'Kruskal-Wallis H': h_statistics,
    'AUC strength': {feature: max(a, 1 - a) for feature, a in roc_results.items()}, # We want the strength of the AUC, not the direction
}).sort_values(['Kruskal-Wallis H', 'AUC strength'], ascending=[False, False])
print(comparison)

'''# Keep only the top features by H statistic (None keeps all of them)
N_SELECTED = None
selected_features = [feature for feature, _ in sorted_features[:N_SELECTED]]

X_train = X_train[selected_features]
X_validation = X_validation[selected_features]
X_test = X_test[selected_features]'''


# --- Data Preparation for Dimensionality Reduction ---
# Calculate standardization parameters on the training set
training_means = X_train.mean()
training_std = X_train.std(ddof=0).replace(0, 1)

# Standardize each set using the same parameters
X_train_processed = (
    (X_train - training_means) / training_std
).to_numpy()

X_validation_processed = (
    (X_validation - training_means) / training_std
).to_numpy()

X_test_processed = (
    (X_test - training_means) / training_std
).to_numpy()

print("\nData preparation:")
print(f"Number of encoded features: {X_train_processed.shape[1]}")


# --- Dimensionality Reduction ---
# PCA
# Center the training data
pca_mean = X_train_processed.mean(axis=0)
X_train_centered = X_train_processed - pca_mean

# Calculate the covariance matrix
covariance_matrix = np.cov(
    X_train_centered,
    rowvar=False,
)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Order the components by decreasing variance
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Remove small negative values caused by numerical rounding
eigenvalues = np.maximum(eigenvalues, 0)

# Calculate the variance explained by each component
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(explained_variance_ratio)

# Select enough components to retain at least 95% of the variance
n_components_pca = int(
    np.searchsorted(cumulative_variance, 0.95) + 1
)

pca_components = eigenvectors[:, :n_components_pca]

# Project all sets using the training mean and components
X_train_pca = X_train_centered @ pca_components

X_validation_pca = (
    X_validation_processed - pca_mean
) @ pca_components

X_test_pca = (
    X_test_processed - pca_mean
) @ pca_components

print("\nPCA results:")
print(f"Original features: {X_train_processed.shape[1]}")
print(f"Selected components: {n_components_pca}")
print(
    f"Explained variance: "
    f"{cumulative_variance[n_components_pca - 1]:.4f}"
)

# Plot the cumulative explained variance
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o',
)

plt.axhline(
    y=0.95,
    color='red',
    linestyle='--',
    label='95% explained variance',
)

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA explained variance')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# LDA
# Select at most one component less than the number of classes
n_components_lda = min(
    len(classes) - 1,
    X_train_processed.shape[1],
)

lda = LinearDiscriminantAnalysis(
    n_components=n_components_lda,
)

# Learn the discriminant directions using the training labels
X_train_lda = lda.fit_transform(
    X_train_processed,
    y_train,
)

# Apply the same transformation to validation and test sets
X_validation_lda = lda.transform(X_validation_processed)
X_test_lda = lda.transform(X_test_processed)

print("\nLDA results:")
print(f"Original features: {X_train_processed.shape[1]}")
print(f"Selected components: {X_train_lda.shape[1]}")
print(f"Explained discriminant ratio: {lda.explained_variance_ratio_}")

# Plot the two discriminant components
plt.figure(figsize=(8, 6))

for class_value in classes:
    class_rows = (y_train == class_value).to_numpy()

    plt.scatter(
        X_train_lda[class_rows, 0],
        X_train_lda[class_rows, 1],
        label=class_value,
        alpha=0.3,
        s=10,
    )

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA projection of the training data')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()