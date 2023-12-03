#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage



# # Data Loading

# In[3]:


data = pd.read_csv('/Users/ninasimone/gene/data.csv')
labels = pd.read_csv('/Users/ninasimone/gene/labels.csv')


# # Merging data with labels

# In[4]:


# Merging data with labels
data['Class'] = labels['Class']


# Splitting data into features and target
X = data.drop(columns=['Class'])
y = data['Class']


# # Data Cleaning

# In[5]:


# Preliminary Analysis
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Removing any rows with missing values for simplicity (you can also impute them)
data = data.dropna()

# Check for duplicates
data = data.drop_duplicates()



# # Distribution of Tumor Types

# In[6]:


# Descriptive Statistics
print(data.describe())

# Class Distribution
print(data['Class'].value_counts())


# Visualizing the distribution of tumor types

# Convert 'Class' to a categorical type
data['Class'] = data['Class'].astype('category')

# Create the countplot with different colors
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
sns.countplot(x='Class', data=data, palette='Set2')  # 'Set2' is an example palette

# Add a title to the plot
plt.title('Distribution of Tumor Types')

# Show the plot
plt.show()


# # Distribution of Gene Expressions

# In[7]:


selected_genes = ['gene_1', 'gene_2', 'gene_3']  # replace with some actual gene names
for gene in selected_genes:
    sns.kdeplot(data=data, x=gene, hue='Class')
    plt.title(f'Distribution of {gene} across tumor types')
    plt.show()


# # Correlation Analysis

# In[8]:


#correlation_matrix = X.corr()
#sns.heatmap(correlation_matrix)
#plt.show()


# Selecting a random subset of genes for correlation analysis
np.random.seed(0)  # for reproducibility
sampled_genes = np.random.choice(X.columns, size=100, replace=False)  # selecting 100 genes randomly

# Calculating the correlation matrix for the sampled genes
sampled_correlation_matrix = data[sampled_genes].corr()

# Plotting the correlation matrix for the sampled genes
plt.figure(figsize=(12, 10))
sns.heatmap(sampled_correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Sampled Gene Expressions')
plt.xlabel('Sampled Genes')
plt.ylabel('Sampled Genes')
plt.show()



# # PCA vs Kernel PCA

# In[9]:


from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns

# Standardize the data
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)



# Assume X_scaled is your scaled feature matrix and y is the categorical class labels from your data
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scale)

# Apply Kernel PCA
kernel_pca = KernelPCA(n_components=2, kernel='rbf') # can try different kernels like 'rbf', 'poly', 'sigmoid'
X_kernel_pca = kernel_pca.fit_transform(X_scale)

# Plot the results of PCA and Kernel PCA with colors based on the categories
plt.figure(figsize=(16, 8))

# PCA Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, legend='full')
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Kernel PCA Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_kernel_pca[:, 0], y=X_kernel_pca[:, 1], hue=y, legend='full')
plt.title('Kernel PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show legend and plot
plt.legend()
plt.tight_layout()
plt.show()


# # Model Evaluation Function

# In[10]:


# Function to evaluate a model
def evaluate_model(model, X, y, is_onehot):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if is_onehot:  # If the target is one-hot encoded, convert predictions for accuracy scoring
        y_test = label_encoder.inverse_transform([np.argmax(y) for y in y_test])
        predictions = label_encoder.inverse_transform([np.argmax(y) for y in predictions])

    return accuracy_score(y_test, predictions)




# # Downsample Data 

# In[11]:


from sklearn.utils import resample

# Separate the dataset into a dictionary with keys as class labels and values as the subset dataframes
class_groups = {cls: X[y == cls] for cls in y.unique()}

# Downsample each class to 78 samples
downsampled_groups = [resample(class_groups[cls], 
                               replace=False,    # sample without replacement
                               n_samples=78,     # to match minority class
                               random_state=42)  # for reproducible results
                     for cls in class_groups]

# Combine the downsampled dataframes
downsampled_data = pd.concat(downsampled_groups)


# Get the new class labels for the downsampled data
downsampled_labels = y.loc[downsampled_data.index]

# Check the new class distribution
print(downsampled_labels.value_counts())

# Now downsampled_data contains your features and downsampled_labels contains your class labels

# Add the labels back to the features
downsampled_data['Class'] = downsampled_labels

# To get the downsampled data without the labels
downsampled_data_without_labels = downsampled_data.drop(columns=['Class'])





# # Tree-based feature Selection

# In[24]:


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(downsampled_data_without_labels)



# One-hot encode the target variable
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(downsampled_labels.values.reshape(-1, 1))


# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(downsampled_labels)


# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_onehot)

importances = rf.feature_importances_

# Selecting features above a certain threshold
threshold = 0.003  # Example threshold
selected_features = downsampled_data_without_labels.columns[importances > threshold]

X_tree_selected = downsampled_data_without_labels[selected_features]
print(X_tree_selected)


#importances = rf.feature_importances_

# Experiment with different thresholds
#thresholds = [0.005, 0.003, 0.001]  # Example thresholds

#for thresh in thresholds:
    #selected_features = X.columns[importances > thresh]
    #X_tree_selected = X[selected_features]
    #print(f"Threshold: {thresh}, Number of Features: {X_tree_selected.shape[1]}")
    #print(X_tree_selected.head())
    




# # Feature importance from Random Forest

# In[42]:


# Filter the feature importances for the selected features only
selected_importances = importances[X.columns.isin(X_tree_selected.columns)]

# Convert the feature importances to a pandas DataFrame
feature_importances_df = pd.DataFrame({
    'Feature': X_tree_selected.columns,
    'Importance': selected_importances
})

# Sort the DataFrame by importance score in descending order
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting the feature importances with a different color palette
plt.figure(figsize=(12, 10))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importances_df.head(20),  # Show top 20 features
    palette=sns.color_palette("hsv", len(feature_importances_df.head(20)))  # Use a hue-saturation-value color palette
)
plt.title('Top 20 Features from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature Names')
plt.tight_layout()  # Fit the plot within the figure neatly
plt.show()



# # Cluster heatmap for top 20 

# In[28]:


# Assuming 'importances' contains importances for all features and 'X' is the original features DataFrame
# First, create a DataFrame mapping features to their importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort by importance and select the top 20 features
top_features_df = feature_importances.sort_values('Importance', ascending=False).head(20)

# Get the names of the top 20 features
top_features = top_features_df['Feature'].values

# Subset your data to include only the top 20 features
X_top_features = X[top_features]


# Perform clustering
row_clusters = linkage(X_top_features.transpose(), method='ward', metric='euclidean')
col_clusters = linkage(X_top_features.T, method='ward', metric='euclidean')

# Create a clustermap
sns.clustermap(X_top_features.transpose(), row_linkage=row_clusters, col_linkage=col_clusters,
               figsize=(12, 12), cmap='viridis')

# Show the plot
plt.show()


# # Model Evaluation for Selected Features (Tree based)

# In[30]:


# Tree-based models with one-hot encoded target
tree_based_models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in tree_based_models.items():
    accuracy = evaluate_model(model, X_tree_selected, y_onehot, is_onehot=True)
    print(f"{name}: {accuracy}")

# Other models with label-encoded target
other_models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

for name, model in other_models.items():
    accuracy = evaluate_model(model, X_tree_selected, y_encoded, is_onehot=False)
    print(f"{name}: {accuracy}")


# # ANOVA Feature Selection

# In[35]:


# Apply ANOVA
#anova_selector = SelectKBest(f_classif, k='all')  # Adjust k as needed
#X_anova_selected = anova_selector.fit_transform(X_scaled, y_encoded)

# Get the p-values for each feature and select based on a threshold
#p_values = anova_selector.pvalues_
#threshold = 0.05  # Example threshold
#selected_features = downsampled_data_without_labels.columns[p_values < threshold]
#X_selected_by_anova =downsampled_data_without_labels[selected_features]

#print(X_selected_by_anova)


# Selecting a fixed number of top features based on ANOVA F-values
num_features = 70  # For example, selecting top 70 features
anova_selector = SelectKBest(f_classif, k=num_features)
X_anova_selected = anova_selector.fit_transform(X_scaled, y_encoded)
selected_features = downsampled_data_without_labels.columns[anova_selector.get_support()]

X_selected_top_features = downsampled_data_without_labels[selected_features]

print(X_selected_top_features)


# # Model Evaluation with ANOVA selected features

# In[36]:


# Tree-based models with one-hot encoded target
tree_based_models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in tree_based_models.items():
    accuracy = evaluate_model(model, X_selected_top_features, y_onehot, is_onehot=True)
    print(f"{name}: {accuracy}")

# Other models with label-encoded target
other_models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

for name, model in other_models.items():
    accuracy = evaluate_model(model, X_selected_top_features, y_encoded, is_onehot=False)
    print(f"{name}: {accuracy}")


# # Feature importance from ANOVA

# In[41]:


# Filter the feature importances for the selected features only
selected_importances = importances[X.columns.isin(X_selected_top_features.columns)]

# Convert the feature importances to a pandas DataFrame
feature_importances_df = pd.DataFrame({
    'Feature': X_selected_top_features.columns,
    'Importance': selected_importances
})

# Sort the DataFrame by importance score in descending order
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting the feature importances with a different color palette
plt.figure(figsize=(12, 10))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importances_df.head(20),  # Show top 20 features
    palette=sns.color_palette("hsv", len(feature_importances_df.head(20)))  # Use a hue-saturation-value color palette
)
plt.title('Top 20 Features from ANOVA')
plt.xlabel('Importance Score')
plt.ylabel('Feature Names')
plt.tight_layout()  # Fit the plot within the figure neatly
plt.show()


# # Column and Row-wise clusterings in heatmap¶

# In[43]:


# Calculate the linkage for rows and columns
row_clusters = linkage(X_selected_top_features.transpose(), method='ward', metric='euclidean')
col_clusters = linkage(X_selected_top_features, method='ward', metric='euclidean')

# Create a clustermap with seaborn
sns.clustermap(X_selected_top_features.transpose(), 
               row_linkage=row_clusters, 
               col_linkage=col_clusters, 
               figsize=(12, 12),  # Adjust this as needed
               cmap='viridis')

# Display the heatmap
plt.show()


# In[ ]:


Further Machine Learning Analysis:

Subgroup Analysis: Use the selected features to identify subgroups within your data. This might reveal distinct patterns or subtypes within different cancer types.
Predictive Modeling: Build predictive models (e.g., for prognosis or treatment response) using these features and evaluate their performance.
External Validation: Apply your model to external datasets to validate the generalizability of your findings.

