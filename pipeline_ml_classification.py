import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Определение признаков
numeric_features = ['num_feature1', 'num_feature2']
categorical_features = ['cat_feature1', 'cat_feature2']
text_features = ['text_feature1', 'text_feature2']

# Трансформеры
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('txt1', text_transformer, text_features[0])
    # Можно добавить остальные текстовые признаки через FeatureUnion или отдельные пайплайны
])

# Модель
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Подготовка данных
X = train_data.drop('target', axis=1)
y = train_data['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Подбор гиперпараметров
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Оценка
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, y_pred))

# Предсказание на тестовой выборке
test_predictions = best_model.predict(test_data)
output = pd.DataFrame({'index': test_data.index, 'prediction': test_predictions})
output.to_csv('predictions.csv', index=False)