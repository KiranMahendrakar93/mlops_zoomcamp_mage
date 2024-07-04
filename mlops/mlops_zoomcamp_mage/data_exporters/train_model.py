import mlflow
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
from scipy.stats import loguniform

# Silence Warnings (optional)
import warnings
warnings.filterwarnings('ignore')

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("mlops_zoomcamp_experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here

    X_train, X_test, y_train, y_test, label_encoders, std_scalers = data

    # Take dump of label encoders
    for col, le in label_encoders.items():
        with open(f'mlops/dumps/misc/{col}_label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)

    # Take dump of standard scaler
    for col, std in std_scalers.items():
        with open(f'mlops/dumps/misc/{col}_standard_scaler.pkl', 'wb') as f:
            pickle.dump(std, f)

    # Dictionary of classification models
    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        #"XGBoost": XGBClassifier(),
        #"CatBoost": CatBoostClassifier(silent=True),
    }

    model_names = []
    accuracies = []

    mlflow.sklearn.autolog()

    # Train and evaluate each model
    for name, clf in classification_models.items():

        with mlflow.start_run():
        
            mlflow.set_tag("developer", "Kiran Mahendrakar")
            mlflow.log_param("model", name)
        
            clf.fit(X_train, y_train)
            train_score = clf.score(X_train, y_train)
            score = clf.score(X_test, y_test)
            model_names.append(name)
            accuracies.append(score)
            print(f"{name} accuracy: {score:.2f}")

            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", score)

            # Save the trained model using pickle
            model_filename = f"{name.replace(' ', '_')}_model.pkl"
            with open(f"mlops/dumps/models/{model_filename}", "wb") as f:
                pickle.dump(clf, f)

            mlflow.sklearn.log_model(sk_model=clf, artifact_path="models")



