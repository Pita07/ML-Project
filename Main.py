import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import auc, roc_curve,  accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# =============================================================================
# --- Load and Prepare the Dataset ---
# =============================================================================

# Get data from the CSV file
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

raw_features = (list(numerical_features) + list(categorical_features))

X_train = data[raw_features].copy()
X_validation = validation_data[raw_features].copy()
X_test = test_data[raw_features].copy()

y_train = data[class_column]
y_validation = validation_data[class_column]
y_test = test_data[class_column]


# --- Missing Values and Categorical Encoding ---
# Represent missing categorical values as a separate category.
# The categories seen in training are used for every set, so that drop_first removes the same dummy column everywhere
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


# =============================================================================
# --- Feature Selection ---
# =============================================================================

# --- Kruskall-Wallis test for numerical features ---
def kruskal_wallis_test(feature):
    # Prepare the data for each class
    values_by_class = [
        X_train.loc[y_train == class_value, feature].dropna().values
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

h_statistics = {feature: kruskal_wallis_test(feature) for feature in X_train.columns}
sorted_features = sorted(h_statistics.items(), key=lambda x: x[1], reverse=True)

'''print("Features ordered by Kruskal-Wallis H statistic:")
for feature, h in sorted_features:
    print(f"{feature}: H = {h:.4f}")'''

# Correlation matrix for the top features
top_features = [feature for feature, _ in sorted_features[:5]]  # Select top 5 features based on H statistic
correlation_matrix = X_train[top_features].corr()

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


# --- ROC Curve and AUC ---
roc_targets = (y_train == "High").astype(int).to_numpy()

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

# Keep only the top features by H statistic (None keeps all of them)
N_SELECTED = None
selected_features = [feature for feature, _ in sorted_features[:N_SELECTED]]

X_train = X_train[selected_features]
X_validation = X_validation[selected_features]
X_test = X_test[selected_features]


# =============================================================================
# ---Dimensionality Reduction ---
# =============================================================================

# --- Data Preparation ---
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


# --- PCA ---
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


# --- LDA ---
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


# =============================================================================
# --- Classification ---
# =============================================================================
# The classifiers are the multi-class version of the ones from the practical class:
# every class k gets a linear score g_k(x) = w_k^T x + b_k and the predicted class is the one with the highest score 

class_labels = np.array(classes)            # sorted class names (order of the score columns)
y_train_array = y_train.to_numpy()
y_validation_array = y_validation.to_numpy()


# --- Minimum Distance (Euclidean linear discriminant) ---
# w_k = m_k  and  b_k = -1/2 m_k^T m_k
def minimum_distance_fit(X, y):
    means = np.array([X[y == class_value].mean(axis=0) for class_value in class_labels])
    W = means                                   # one row w_k per class
    b = -0.5 * np.sum(means * means, axis=1)    # b_k = -1/2 m_k^T m_k
    return W, b

def minimum_distance_scores(X, W, b):
    return X @ W.T + b                          # column k holds g_k(x)


# --- Fisher Linear Discriminant ---
# With K classes there are K-1 Fisher directions: the leading eigenvectors of SW^-1 SB
# A sample is assigned to the class with the closest mean in the projected space
def fisher_fit(X, y):
    overall_mean = X.mean(axis=0)
    n_features = X.shape[1]

    SW = np.zeros((n_features, n_features))     # within-class scatter
    SB = np.zeros((n_features, n_features))     # between-class scatter
    for class_value in class_labels:
        X_class = X[y == class_value]
        class_mean = X_class.mean(axis=0)

        centered = X_class - class_mean
        SW += centered.T @ centered

        mean_difference = (class_mean - overall_mean).reshape(-1, 1)
        SB += len(X_class) * (mean_difference @ mean_difference.T)

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW) @ SB)
    order = np.argsort(eigenvalues.real)[::-1]
    n_directions = min(len(class_labels) - 1, n_features)
    W = eigenvectors[:, order[:n_directions]].real

    # Scale each direction so that w^T SW w = 1. Then the Euclidean distance in the
    # projected space weights every direction by its within-class spread
    W = W / np.sqrt(np.sum(W * (SW @ W), axis=0))

    projected_means = np.array([
        X[y == class_value].mean(axis=0) @ W for class_value in class_labels
    ])
    return W, projected_means

def fisher_scores(X, W, projected_means):
    projected = X @ W
    squared_distances = np.sum(
        (projected[:, None, :] - projected_means[None, :, :]) ** 2, axis=2
    )
    return -squared_distances                   # closest mean = highest score


# --- Bayes Classifier (one full-covariance Gaussian per class) ---
# Discriminant: g_k(x) = ln p(x | class k) + ln P(class k); the predicted class has the highest g_k.
# Logarithms are used because with many features the densities p(x | class k) underflow to 0.
def bayes_fit(X, y):
    means, covariances, priors = {}, {}, {}
    for class_value in class_labels:
        class_samples = X[y == class_value]

        gaussian = GaussianMixture(
            n_components=1,
            covariance_type='full',
            random_state=0,
        ).fit(class_samples)
        means[class_value] = gaussian.means_[0]
        covariances[class_value] = gaussian.covariances_[0]
        priors[class_value] = class_samples.shape[0] / X.shape[0]
    return means, covariances, priors

def bayes_scores(X, means, covariances, priors):
    d = X.shape[1]
    scores = np.zeros((X.shape[0], len(class_labels)))
    for k, class_value in enumerate(class_labels):
        # The inverse and the determinant are computed once per class, not once per sample
        covariance_inverse = np.linalg.inv(covariances[class_value])
        _, log_determinant = np.linalg.slogdet(covariances[class_value])
        centered = X - means[class_value]
        mahalanobis_squared = np.sum((centered @ covariance_inverse) * centered, axis=1)

        scores[:, k] = (
            -0.5 * mahalanobis_squared
            - 0.5 * log_determinant
            - 0.5 * d * np.log(2 * np.pi)
            + np.log(priors[class_value])
        )
    return scores


# --- kNN Classifier ---
def knn_fit(X, y, k):
    return KNeighborsClassifier(n_neighbors=k).fit(X, y)

def knn_scores(X, knn_model):
    # Fraction of the k neighbours that belong to each class
    # (the columns follow knn_model.classes_, which is sorted like class_labels)
    return knn_model.predict_proba(X)

# k is a hyper-parameter, so it is chosen on the validation set (odd values only)
k_values = list(range(1, 30, 2))

def choose_k(X_tr, y_tr, X_val, y_val):
    validation_accuracy = {}
    for k in k_values:
        knn_model = knn_fit(X_tr, y_tr, k)
        y_pred = class_labels[np.argmax(knn_scores(X_val, knn_model), axis=1)]
        validation_accuracy[k] = np.mean(y_pred == y_val)
    best_k = max(validation_accuracy, key=validation_accuracy.get)  # ties keep the smallest k
    return best_k, validation_accuracy


# --- SVM Classifier ---
# Linear kernel: linear decision boundaries
# RBF kernel: non-linear decision boundaries

def svm_fit(X, y, kernel):
    return SVC(
        kernel=kernel,
        C=1.0,
        gamma='scale',
    ).fit(X, y)


# --- Decision Tree Classifier ---

def decision_tree_fit(X, y):
    return DecisionTreeClassifier(
        random_state=42,
    ).fit(X, y)


# --- Random Forest Classifier ---

def random_forest_fit(X, y):
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ).fit(X, y)


# --- AdaBoost Classifier ---

def adaboost_fit(X, y):
    return AdaBoostClassifier(
        n_estimators=50,
        random_state=42,
    ).fit(X, y)


# --- Run the classifiers ---
# Validation is used to compare pipelines; the test set is left untouched for the final assessment.
feature_sets = {
    '\nOriginal': (X_train_processed, X_validation_processed),
    'PCA': (X_train_pca, X_validation_pca),
    'LDA': (X_train_lda, X_validation_lda),
}

for set_name, (X_tr, X_val) in feature_sets.items():
    # Minimum Distance
    W_md, b_md = minimum_distance_fit(X_tr, y_train_array)
    y_pred_md = class_labels[np.argmax(minimum_distance_scores(X_val, W_md, b_md), axis=1)]

    # Fisher LDA
    W_f, projected_means = fisher_fit(X_tr, y_train_array)
    y_pred_f = class_labels[np.argmax(fisher_scores(X_val, W_f, projected_means), axis=1)]

    # Bayes
    means, covariances, priors = bayes_fit(X_tr, y_train_array)
    y_pred_bayes = class_labels[np.argmax(bayes_scores(X_val, means, covariances, priors), axis=1)]

    # kNN
    best_k, _ = choose_k(X_tr, y_train_array, X_val, y_validation_array)
    knn_model = knn_fit(X_tr, y_train_array, best_k)
    y_pred_knn = class_labels[np.argmax(knn_scores(X_val, knn_model), axis=1)]

    # Linear SVM
    svm_linear_model = svm_fit(
        X_tr,
        y_train_array,
        kernel='linear',
    )

    y_pred_svm_linear = svm_linear_model.predict(X_val)

    # Non-linear SVM
    svm_rbf_model = svm_fit(
        X_tr,
        y_train_array,
        kernel='rbf',
    )

    y_pred_svm_rbf = svm_rbf_model.predict(X_val)

    # Decision Tree
    decision_tree_model = decision_tree_fit(X_tr, y_train_array)
    y_pred_tree = decision_tree_model.predict(X_val)

    # Random Forest
    random_forest_model = random_forest_fit(X_tr, y_train_array)
    y_pred_forest = random_forest_model.predict(X_val)

    # AdaBoost
    adaboost_model = adaboost_fit(X_tr, y_train_array)
    y_pred_adaboost = adaboost_model.predict(X_val)

    # Temporary print before doing 3.5
    print(
        f"{set_name}: validation accuracy"
        f" | Minimum Distance = {np.mean(y_pred_md == y_validation_array):.4f}"
        f" | Fisher LDA = {np.mean(y_pred_f == y_validation_array):.4f}"
        f" | Bayes = {np.mean(y_pred_bayes == y_validation_array):.4f}"
        f" | kNN (k={best_k}) = {np.mean(y_pred_knn == y_validation_array):.4f}"
        f" | Linear SVM = {np.mean(y_pred_svm_linear == y_validation_array):.4f}"
        f" | RBF SVM = {np.mean(y_pred_svm_rbf == y_validation_array):.4f}"
        f" | Decision Tree = {np.mean(y_pred_tree == y_validation_array):.4f}"
        f" | Random Forest = {np.mean(y_pred_forest == y_validation_array):.4f}"
        f" | AdaBoost = {np.mean(y_pred_adaboost == y_validation_array):.4f}"
    )
