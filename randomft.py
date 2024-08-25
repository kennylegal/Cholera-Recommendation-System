import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving and loading models

# Define a custom tokenizer function (replacing the lambda function)
def custom_tokenizer(text):
    return text

# Step 1: Load and Preprocess the Data
data = pd.read_csv('cholera_symptoms_treatment.csv')

# Split the 'Symptoms' and 'Treatment' into lists
data['Symptoms'] = data['Symptoms'].apply(lambda x: x.split(', '))
data['Treatment'] = data['Treatment'].apply(lambda x: x.split(', '))

# Convert the list of symptoms into a binary matrix
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)
X = vectorizer.fit_transform(data['Symptoms']).toarray()

# Convert the list of treatments into a binary matrix
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['Treatment'])

# Step 2: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

# Step 4: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters from Grid Search:", best_params)

# Step 5: Train the Random Forest Model with Best Parameters
rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(X_train, y_train)

# Step 6: Predict on the Test Set
y_pred = rf_best.predict(X_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=mlb.classes_)
print("Classification Report:\n", report)

# Step 8: Feature Importance Analysis
importances = rf_best.feature_importances_
feature_names = vectorizer.get_feature_names_out()

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Symptoms')
plt.show()

#  Saving the trained Model and Preprocessing Objects
# Save the trained model
# joblib.dump(rf_best, 'MyRecommendationModel.pkl')

# # Saving the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Saving the label binarizer
# joblib.dump(mlb, 'label_binarizer.pkl')

print("Model and preprocessing objects have been saved to disk.")

# Step 10: Loading the Model and Preprocessing Objects (for later use)
# To load the model and preprocessing objects in another script, use the following:
# rf_loaded = joblib.load('random_forest_model.pkl')
# vectorizer_loaded = joblib.load('vectorizer.pkl')
# mlb_loaded = joblib.load('label_binarizer.pkl')
