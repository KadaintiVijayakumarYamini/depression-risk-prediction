import pandas as pd

df = pd.read_csv(r"C:\Users\yamin\OneDrive\student depression risk predictor\Student Depression Dataset.csv")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nClass balance:\n", df["Depression"].value_counts(normalize=True))
df.info()
print(df.head())


# remove stray quotes and extra spaces from text columns
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.replace("'", "", regex=False).str.strip()

# shorter name for the long column
df = df.rename(columns={"Have you ever had suicidal thoughts ?": "Suicidal Thoughts"})
# 
# %

# check the categories in each text column
for col in ["Gender", "Profession", "Sleep Duration", "Dietary Habits",
            "Suicidal Thoughts", "Family History of Mental Illness"]:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())

print("\nDegree unique values:", df["Degree"].nunique())
print("City unique values:", df["City"].nunique())

# check the columns that may be almost constant
print("\nWork Pressure:\n", df["Work Pressure"].value_counts())
print("\nJob Satisfaction:\n", df["Job Satisfaction"].value_counts())

# 1. drop columns that add little or nothing
df = df.drop(columns=["id", "Profession", "Work Pressure", "Job Satisfaction", "City"])

# 2. remove the rare "Others" rows and the 3 missing Financial Stress rows
df = df[(df["Sleep Duration"] != "Others") & (df["Dietary Habits"] != "Others")]
df = df.dropna(subset=["Financial Stress"])

# 3. ordered columns -> numbers
sleep_map = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
diet_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map)
df["Dietary Habits"] = df["Dietary Habits"].map(diet_map)

# 4. yes/no and gender columns -> 0/1
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Suicidal Thoughts"] = df["Suicidal Thoughts"].map({"Yes": 1, "No": 0})
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

# 5. check the result
print("New shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nDegree top 10:\n", df["Degree"].value_counts().head(10))

import matplotlib.pyplot as plt
import seaborn as sns

# 1. class balance
sns.countplot(x="Depression", data=df)
plt.title("Class balance (0 = No, 1 = Yes)")
plt.show()

# 2. depression rate by key features
for col in ["Sleep Duration", "Dietary Habits", "Suicidal Thoughts",
            "Family History of Mental Illness", "Financial Stress",
            "Academic Pressure", "Study Satisfaction", "Gender"]:
    df.groupby(col)["Depression"].mean().plot(kind="bar")
    plt.title(f"Depression rate by {col}")
    plt.ylabel("Depression rate")
    plt.tight_layout()
    plt.show()

# 3. distributions by depression
for col in ["Age", "CGPA", "Work/Study Hours"]:
    sns.histplot(data=df, x=col, hue="Depression", kde=True, bins=30)
    plt.title(f"{col} by Depression")
    plt.show()

# 4. correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=["Degree"]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.show()


corr = df.drop(columns=["Degree"]).corr()["Depression"].sort_values(ascending=False)
print(corr)

from sklearn.model_selection import train_test_split

# 1. keep the 10 most common degrees, group the rest as "Other"
top_degrees = df["Degree"].value_counts().head(10).index
df["Degree"] = df["Degree"].where(df["Degree"].isin(top_degrees), "Other")

# 2. one-hot encode Degree (text -> 0/1 columns)
df = pd.get_dummies(df, columns=["Degree"], dtype=int)

# 3. separate features (X) and target (y)
X = df.drop(columns=["Depression"])
y = df["Depression"]

# 4. split 80% train / 20% test, keeping the class ratio the same in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 5. check the result
print("X shape:", X.shape)
print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("\nTrain class balance:\n", y_train.value_counts(normalize=True))
print("\nTest class balance:\n", y_test.value_counts(normalize=True))
print("\nColumns:", list(X.columns))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# train a tree with no restrictions
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# compare accuracy on training data vs unseen test data
train_acc = accuracy_score(y_train, tree.predict(X_train))
test_acc = accuracy_score(y_test, tree.predict(X_test))

print("Train accuracy:", round(train_acc, 4))
print("Test accuracy :", round(test_acc, 4))
print("Gap           :", round(train_acc - test_acc, 4))
print("Tree depth    :", tree.get_depth())
print("Number of leaves:", tree.get_n_leaves())


from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. settings to try
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 20, 50],
    "min_samples_leaf": [1, 10, 30],
    "ccp_alpha": [0.0, 0.0005, 0.001],
}

# 2. 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. search (F1 balances precision and recall)
grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

# 4. results
best_tree = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best CV F1     :", round(grid.best_score_, 4))

train_acc = accuracy_score(y_train, best_tree.predict(X_train))
test_acc = accuracy_score(y_test, best_tree.predict(X_test))
print("Train accuracy :", round(train_acc, 4))
print("Test accuracy  :", round(test_acc, 4))
print("Gap            :", round(train_acc - test_acc, 4))
print("Tree depth     :", best_tree.get_depth())
print("Number of leaves:", best_tree.get_n_leaves())

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

y_pred = best_tree.predict(X_test)
y_prob = best_tree.predict_proba(X_test)[:, 1]

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))
print("\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["No", "Yes"]).plot(cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

# 1. feature importances
imp = pd.Series(best_tree.feature_importances_, index=X.columns)
imp = imp[imp > 0].sort_values()
print(imp.sort_values(ascending=False))

imp.plot(kind="barh", figsize=(8, 5), title="Feature importance (Decision Tree)")
plt.tight_layout()
plt.show()

# 2. plot the tree
plt.figure(figsize=(22, 10))
plot_tree(best_tree, feature_names=X.columns, class_names=["No", "Yes"],
          filled=True, fontsize=10)
plt.show()

# 3. the same tree as text rules

print(export_text(best_tree, feature_names=list(X.columns)))


import joblib

joblib.dump(best_tree, "depression_tree_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("Saved model and column list")


import os
print(os.getcwd())
