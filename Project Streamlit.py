# To open the interface, run 'streamlit run Project Streamlit.py' in the terminal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import streamlit as st

# Load your dataset here
@st.cache_data
def load_data():
    # Placeholder: replace with your dataset loading logic
    df = pd.read_csv("OnlineNewsPopularity.csv")
    df.drop(columns=['url'], inplace=True)  # Example of dropping a column
    return df

df = load_data()


def preprocess_data(df, feature_selection_method):
    # Assuming 'shares' is the target and applying discretization as per your setup
    bins = [0, 10000, 100000, 1000000]
    labels = [0, 1, 2]
    df['shares_class'] = pd.cut(df[' shares'], bins=bins, labels=labels, include_lowest=True)
    df.drop(columns=[' shares'], inplace=True)

    # Feature and target separation
    X = df.drop('shares_class', axis=1)
    y = df['shares_class']

    # Feature scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Handling imbalance
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_scaled, y)

    # Feature selection
    if feature_selection_method == 'PCA':
        pca = PCA(n_components=10)
        X_transformed = pca.fit_transform(X_sm)
    elif feature_selection_method == 'LDA':
        lda = LDA(n_components=2)
        X_transformed = lda.fit_transform(X_sm, y_sm)
    
    return X_transformed, y_sm


# Train model
def train_model(X, y, classifier_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if classifier_type == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    elif classifier_type == 'Decision Tree':
        classifier = DecisionTreeClassifier(random_state = 42)
    elif classifier_type == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors = 6)
    elif classifier_type == 'CatBoost':
        classifier = CatBoostClassifier(iterations=1000, learning_rate=0.1, loss_function='MultiClass', eval_metric='Accuracy', random_seed=42, verbose=100)
    elif classifier_type == 'XGBoost':
        classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)), random_state=42, n_estimators=150, max_depth=10,)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    return y_test, y_pred, y_proba, classifier

# Plotting functions
def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_roc_curve(y_test, y_proba):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_proba[:, i])

    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit UI
st.title('Online News Popularity Classification App')

feature_selection_method = st.selectbox("Feature Selection Method", ["PCA", "LDA"])
classifier_type = st.selectbox("Classifier Type", ["Random Forest", "Decision Tree", "KNN", "CatBoost", "XGBoost"])

if st.button('Train & Evaluate'):
    X, y = preprocess_data(df, feature_selection_method)
    y_test, y_pred, y_proba, classifier = train_model(X, y, classifier_type)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"Accuracy: {accuracy}")
    st.text("Classification Report")
    st.text(report)

    st.write("Confusion Matrix")
    plot_confusion_matrix(cm)

    st.write("ROC Curve")
    plot_roc_curve(y_test, y_proba)