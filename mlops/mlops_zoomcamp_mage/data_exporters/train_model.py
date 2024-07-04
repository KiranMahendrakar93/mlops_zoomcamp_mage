import mlflow
import pickle

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

    X_train, X_test, y_train, y_test = data

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
            with open(f"/dumps/models/{model_filename}", "wb") as f:
                pickle.dump(clf, f)

            mlflow.sklearn.log_model(sk_model=clf, artifact_path="models")



