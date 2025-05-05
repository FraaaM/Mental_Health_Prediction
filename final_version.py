import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def process_sleep_duration(df):
    def range_to_mean(range_str):
        range_str = range_str.replace(' hours', '').strip()
        if '-' in range_str:
            start, end = map(float, range_str.split('-'))
            return (start + end) / 2, start, end
        elif range_str == 'Less than 5':
            return 4.0, 4.0, 4.0
        elif range_str == 'More than 8':
            return 9.0, 9.0, 9.0
        else:
            return float(range_str), float(range_str), float(range_str)
    
    df[['Sleep_Mean', 'Sleep_Min', 'Sleep_Max']] = df['Sleep Duration'].apply(
        lambda x: pd.Series(range_to_mean(x))
    )
    return df.drop('Sleep Duration', axis=1)

def process_dietary_habits(df):
    dietary_order = [['Unhealthy', 'Moderate', 'Healthy']]
    encoder = OrdinalEncoder(categories=dietary_order)
    df['Dietary_Habits_Ordinal'] = encoder.fit_transform(df[['Dietary Habits']])
    return df

def create_interaction_features(df):
    df['Work_Study_Stress'] = df['Work/Study Hours'] * (
        df['Academic Pressure'].fillna(0) + df['Work Pressure'].fillna(0)
    )
    return df

# Функция для категоризации возраста
def bin_age(df):
    bins = [0, 25, 35, 45, 60, 100]
    labels = ['Молодой', 'Молодой_взрослый', 'Взрослый', 'Средний_возраст', 'Пожилой']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    return df

def preprocess_data(train, test):
    train = train.drop("Name", axis=1)
    test = test.drop("Name", axis=1)
    
    # Бинарное кодирование
    for col, mapping in [
        ("Gender", {"Male": 0, "Female": 1}),
        ("Have you ever had suicidal thoughts ?", {"No": 0, "Yes": 1}),
        ("Family History of Mental Illness", {"No": 0, "Yes": 1})
    ]:
        train[col] = train[col].map(mapping)
        test[col] = test[col].map(mapping)
    
    train = train.rename(columns={
        "Have you ever had suicidal thoughts ?": "Suicidal_Thoughts",
        "Family History of Mental Illness": "Family_History"
    })
    test = test.rename(columns={
        "Have you ever had suicidal thoughts ?": "Suicidal_Thoughts",
        "Family History of Mental Illness": "Family_History"
    })
    
    train["Is_Student"] = (train["Working Professional or Student"] == "Student").astype(int)
    test["Is_Student"] = (test["Working Professional or Student"] == "Student").astype(int)
    
    train.loc[train["Is_Student"] == 1, ["Work Pressure", "Job Satisfaction"]] = 0
    test.loc[test["Is_Student"] == 1, ["Work Pressure", "Job Satisfaction"]] = 0
    
    train.loc[train["Is_Student"] == 0, ["Academic Pressure", "CGPA", "Study Satisfaction"]] = 0
    test.loc[test["Is_Student"] == 0, ["Academic Pressure", "CGPA", "Study Satisfaction"]] = 0
    
    # Частотное кодирование для City, Profession, Degree
    for col in ["City", "Profession", "Degree"]:
        freq = train[col].value_counts()
        rare_categories = freq[freq < 10].index
        train[col] = train[col].replace(rare_categories, 'Другое')
        test[col] = test[col].replace(rare_categories, 'Другое')
        
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(freq.mean())
    
    train = train.drop(["City", "Profession", "Degree"], axis=1)
    test = test.drop(["City", "Profession", "Degree"], axis=1)
    
    train = train.drop("Working Professional or Student", axis=1)
    test = test.drop("Working Professional or Student", axis=1)
    
    train = process_sleep_duration(train)
    test = process_sleep_duration(test)
    
    train = process_dietary_habits(train)
    test = process_dietary_habits(test)
    
    train = create_interaction_features(train)
    test = create_interaction_features(test)
    
    train = bin_age(train)
    test = bin_age(test)
    
    return train, test

train, test = preprocess_data(train, test)

numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ["Depression", "id"]]

categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num Bitesize', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = LogisticRegression()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Подбор гиперпараметров
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__penalty': ['l2']  # l1 только для liblinear
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=skf,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(train.drop(["Depression", "id"], axis=1), train["Depression"])

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая F1-оценка: {grid_search.best_score_:.4f}")

# Оценка модели
scoring = {'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
scores = cross_validate(
    grid_search.best_estimator_,
    train.drop(["Depression", "id"], axis=1),
    train["Depression"],
    cv=skf,
    scoring=scoring,
    return_train_score=False
)

print("Логистическая регрессия:")
print(f"  F1-оценка = {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std() * 2:.4f})")
print(f"  Точность = {scores['test_precision'].mean():.4f}")
print(f"  Полнота = {scores['test_recall'].mean():.4f}\n")

best_model = grid_search.best_estimator_
best_model.fit(train.drop(["Depression", "id"], axis=1), train["Depression"])

predictions = best_model.predict(test.drop("id", axis=1))

submission = pd.DataFrame({"id": test["id"], "Depression": predictions})
submission.to_csv("submission.csv", index=False)
print("Файл submission.csv успешно создан!")