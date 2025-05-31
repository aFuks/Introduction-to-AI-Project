# Import niezbednych bibliotek
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math

from sklearn.model_selection import GridSearchCV
def tune_hyperparameters(classifiers, param_grids, X_train, y_train, cv_folds):
    best_classifiers = {}
    for name, clf_info in classifiers.items():
        print(f"\nPrzeszukiwanie siatki hiperparametrów dla: {name}")
        base_model = clf_info['model']()

        param_grid = param_grids.get(name, {})
        if not param_grid:
            print("  ⚠️  Brak siatki parametrów – używam domyślnych ustawień.")
            base_model.set_params(**clf_info['params'])
            best_classifiers[name] = base_model
            continue

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv_folds,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"  ✅ Najlepsze parametry: {grid_search.best_params_}")
        best_classifiers[name] = best_model
    return best_classifiers


param_grids = {
    'k-NN': {
        'n_neighbors': [3, 5, 7, 9]
    },
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    'SVM (RBF)': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'Decision Tree (J48)': {
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }
}


warnings.filterwarnings('ignore')

# --------------------- PARAMETRY ---------------------
params = {
    'dataset_id': 62,                    # ID zbioru danych w UCI repo
    'test_size': 0.3,                   # Proporcja danych testowych
    'random_state': 42,                 # Losowosc podzialu i modeli
    'stratify': True,                   # Czy stosowac stratifikacje
    
    # Imputer
    'use_imputer': True,
    'imputer_strategy': 'most_frequent',  # 'mean', 'median', 'most_frequent', 'constant'
    
    # Skalowanie: 'standard', 'minmax', None
    'scaler': 'standard',
    
    # Klasyfikatory i ich parametry
    'classifiers': {
        'k-NN': {'model': KNeighborsClassifier, 'params': {'n_neighbors': 5}},
        'Naive Bayes': {'model': GaussianNB, 'params': {}},
        'SVM (RBF)': {'model': SVC, 'params': {'kernel': 'poly', 'random_state': 42}},
        'Decision Tree (J48)': {'model': DecisionTreeClassifier, 'params': {'random_state': 42}},
        'Random Forest': {'model': RandomForestClassifier, 'params': {'n_estimators': 100, 'random_state': 42}},
        'Gradient Boosting': {'model': GradientBoostingClassifier, 'params': {'n_estimators': 100, 'random_state': 42}}
    },
    
    'cross_val_folds': 7,               # Liczba foldow w walidacji krzyzowej
    
    # Co pokazac/wykreslic
    'show_plots': True,
    'show_reports': True,
    
    # Zapis do pliku
    'save_results': True,
    'results_filename': 'wyniki_klasyfikacji.txt'
}
# ------------------------------------------------------

# Pobranie zbioru danych o raku pluc
print("Pobieranie zbioru danych o raku pluc z UCI Repository...")
lung_cancer = fetch_ucirepo(id=params['dataset_id'])

# Przygotowanie danych
X = lung_cancer.data.features
y = lung_cancer.data.targets

print(f"Kształt danych: {X.shape}")
print(f"Liczba klas: {y.nunique().values[0]}")
print(f"Rozklad klas:")
print(y.value_counts())

def preprocess_data(X, y, use_imputer, imputer_strategy, scaler_type):
    """
    Funkcja do przetwarzania wstepnego danych
    """
    X_processed = X.copy()
    y_processed = y.copy()

    if use_imputer:
        imputer = SimpleImputer(strategy=imputer_strategy)
        X_processed = pd.DataFrame(imputer.fit_transform(X_processed), columns=X_processed.columns)

    le = LabelEncoder()
    y_processed = le.fit_transform(y_processed.values.ravel())

    if scaler_type == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
    else:
        X_scaled = X_processed

    return X_scaled, y_processed, le

# Przetworzenie danych
X_processed, y_processed, label_encoder = preprocess_data(
    X, y,
    params['use_imputer'],
    params['imputer_strategy'],
    params['scaler']
)

# Podzial na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed,
    test_size=params['test_size'],
    random_state=params['random_state'],
    stratify=y_processed if params['stratify'] else None
)

print(f"Rozmiar zbioru treningowego: {X_train.shape}")
print(f"Rozmiar zbioru testowego: {X_test.shape}")

# Utworzenie obiektow klasyfikatorow
classifiers = tune_hyperparameters(
    classifiers=params['classifiers'],
    param_grids=param_grids,
    X_train=X_train,
    y_train=y_train,
    cv_folds=params['cross_val_folds']
)

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, clf_name):
    """
    Funkcja do ewaluacji klasyfikatora
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'Classifier': clf_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Predictions': y_pred
    }
    return results

# Ewaluacja klasyfikatorow
results = {}
for name, clf in classifiers.items():
    print(f"\nEwaluacja klasyfikatora: {name}")
    result = evaluate_classifier(clf, X_train, X_test, y_train, y_test, name)
    results[name] = result
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1-Score: {result['F1-Score']:.4f}")

# Podsumowanie wynikow
results_df = pd.DataFrame([
    {
        'Klasyfikator': res['Classifier'],
        'Accuracy': res['Accuracy'],
        'Precision': res['Precision'],
        'Recall': res['Recall'],
        'F1-Score': res['F1-Score']
    } for res in results.values()
])

print("\n" + "="*70)
print("PODSUMOWANIE WYNIKOW KLASYFIKACJI")
print("="*70)
print(results_df.round(4).to_string(index=False))

if params['show_plots']:
    # Wykres porownujacy metryki
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, metric in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        bars = ax.bar(results_df['Klasyfikator'], results_df[metric],
                      color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
        ax.set_title(f'Porownanie {metric}', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(0, 1.1)

        for bar, value in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.suptitle('Porownanie wydajnosci klasyfikatorow na zbiorze danych o raku pluc',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()

    # Macierze pomylek z automatycznym dostosowaniem rozmiaru subplotow
    n_classifiers = len(results)
    n_cols = 2
    n_rows = math.ceil(n_classifiers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    # Upewnij sie, ze axes jest 2D tablica nawet gdy jest 1 rzad lub 1 kolumna
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    class_names = [str(c) for c in label_encoder.classes_]

    for idx, (name, result) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        cm = result['Confusion Matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Macierz pomylek - {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Przewidywane', fontsize=12)
        ax.set_ylabel('Rzeczywiste', fontsize=12)

    # Ukryj puste subploty, jesli sa
    for i in range(n_classifiers, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.suptitle('Macierze pomylek dla wszystkich klasyfikatorow',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()

if params['show_reports']:
    print("\n" + "="*70)
    print("SZCZEGOLowe RAPORTY KLASYFIKACJI")
    print("="*70)
    target_names = [str(c) for c in label_encoder.classes_]

    for name, result in results.items():
        print(f"\n{name}:")
        print("-" * 50)
        report = classification_report(y_test, result['Predictions'], target_names=target_names)
        print(report)

print("\n" + "="*70)
print("WALIDACJA KRZYZOWA")
print("="*70)

cv_results = {}
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_processed, y_processed,
                                cv=params['cross_val_folds'], scoring='accuracy')
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if params['show_plots']:
    plt.figure(figsize=(12, 8))
    cv_means = [cv_results[name]['mean'] for name in classifiers.keys()]
    cv_stds = [cv_results[name]['std'] for name in classifiers.keys()]

    bars = plt.bar(range(len(classifiers)), cv_means, yerr=cv_stds,
                   capsize=5, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'],
                   alpha=0.8)

    plt.xlabel('Klasyfikatory', fontsize=12)
    plt.ylabel('Accuracy (Walidacja krzyzowa)', fontsize=12)
    plt.title('Wyniki walidacji krzyzowej (5-fold) dla wszystkich klasyfikatorow',
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(classifiers)), classifiers.keys(), rotation=45)
    plt.ylim(0, 1.1)

    for bar, mean, std in zip(bars, cv_means, cv_stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                 f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_results_to_file(results, cv_results, filename):
    """
    Zapisuje wyniki do pliku tekstowego
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("WYNIKI KLASYFIKACJI DANYCH O RAKU PŁUC\n")
        f.write("="*50 + "\n\n")

        f.write("Parametry zbioru danych:\n")
        f.write(f"- Liczba probek: {X.shape[0]}\n")
        f.write(f"- Liczba cech: {X.shape[1]}\n")
        f.write(f"- Liczba klas: {len(label_encoder.classes_)}\n\n")

        f.write("Wyniki ewaluacji na zbiorze testowym:\n")
        f.write("-" * 40 + "\n")
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy:  {result['Accuracy']:.4f}\n")
            f.write(f"  Precision: {result['Precision']:.4f}\n")
            f.write(f"  Recall:    {result['Recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['F1-Score']:.4f}\n")

        f.write("\n\nWyniki walidacji krzyzowej:\n")
        f.write("-" * 40 + "\n")
        for name, cv_result in cv_results.items():
            f.write(f"{name}: {cv_result['mean']:.4f} (+/- {cv_result['std'] * 2:.4f})\n")

if params['save_results']:
    save_results_to_file(results, cv_results, params['results_filename'])
    print(f"\nWyniki zostaly zapisane do pliku '{params['results_filename']}'")

print("\n" + "="*70)
print("ANALIZA ZAKONCZONA")
print("="*70)

print("\n" + "="*70)
print("ANALIZA STATYSTYCZNA I WNIOSKI")
print("="*70)

best_accuracy = max(results.items(), key=lambda x: x[1]['Accuracy'])
best_f1 = max(results.items(), key=lambda x: x[1]['F1-Score'])
best_cv = max(cv_results.items(), key=lambda x: x[1]['mean'])

print(f"\nNajlepsze wyniki:")
print(f"- Najwyzsza dokladnosc: {best_accuracy[0]} ({best_accuracy[1]['Accuracy']:.4f})")
print(f"- Najwyzszy F1-Score: {best_f1[0]} ({best_f1[1]['F1-Score']:.4f})")
print(f"- Najlepsza walidacja krzyzowa: {best_cv[0]} ({best_cv[1]['mean']:.4f})")

print(f"\nCharakterystyka zbioru danych:")
print(f"- Stosunek cech do probek: {X.shape[1]}/{X.shape[0]} = {X.shape[1]/X.shape[0]:.2f}")
print(f"- Maly zbior danych moze prowadzic do overfittingu")
print(f"- Wysoka wymiarowosc wymaga ostroznej interpretacji wynikow")

print(f"\nRekomendacje:")
print(f"- Rozwazenie redukcji wymiarowosci (PCA, feature selection)")
print(f"- Zastosowanie regularyzacji w modelach")
print(f"- Zwiekszenie rozmiaru zbioru danych jesli to mozliwe")
print(f"- Uzycie ensemblingu dla poprawy stabilnosci wynikow")
