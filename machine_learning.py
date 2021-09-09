from math import log
import numpy as np
import pandas as pd
from scipy.sparse.construct import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


def data_imputation(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(labels='Loan_ID', axis=1, inplace=True)

    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    df['Education'] = df['Education'].map({'Not Graduate': 0, 'Graduate': 1})
    df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 0, 'Semiurban': 1, 'Rural': 2})
    df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

    # Data Imputation
    mode = df.mode()

    df['Gender'].fillna(value=int(mode['Gender']), inplace=True)
    df['Married'].fillna(value=int(mode['Married']), inplace=True)
    df['Dependents'].fillna(value=int(mode['Dependents']), inplace=True)
    df['Self_Employed'].fillna(value=int(mode['Self_Employed']), inplace=True)
    df['Credit_History'].fillna(value=int(mode['Credit_History']), inplace=True)

    df['LoanAmount'].fillna(value=df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(value=df['Loan_Amount_Term'].median(), inplace=True)

    return df


def preprocess_data(df):
    # Outlier elimination
    mu = np.mean(df, axis=0)[:-1]
    sigma = np.std(df, axis=0)[:-1]

    # mu - 2sigma and mu + 2sigma
    lower_bound = np.array([mu[i] - (2 * sigma[i]) for i in range(len(mu))])
    upper_bound = np.array([mu[i] + (2 * sigma[i]) for i in range(len(mu))])

    X = df.drop(labels='Loan_Status', axis=1, inplace=False)
    y = df['Loan_Status']

    cleaned = []
    cleaned_labels = []
    for i, row in enumerate(X.values):
        counter = 0
        for j, col in enumerate(row):
            if lower_bound[j] < col < upper_bound[j]:
                counter += 1

        if counter > 7:
            cleaned.append(row)
            cleaned_labels.append(y[[i]])

    X = np.array(cleaned)
    y = np.array(cleaned_labels)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    yhat = lof.fit_predict(X)
    inliers = yhat != -1

    X = X[inliers, :]
    y = y[inliers].ravel()

    return X, y


def split_and_normalize(X, y):
    X_, x_test, y_, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    normalizer = MinMaxScaler().fit(X_)
    x_norm = normalizer.transform(X_)
    x_test_norm = normalizer.transform(x_test)
    x_train, x_cv, y_train, y_cv = train_test_split(x_norm, y_, test_size=0.2, random_state=42)

    print(f"[Training] Data shape: {x_train.shape}, labels: {y_train.shape}")
    print(f"[Cross-validation] Data shape: {x_cv.shape}, labels: {y_cv.shape}")
    print(f"[Testing] Data shape: {x_test_norm.shape}, labels: {y_test.shape}\n")

    return x_norm, y_, x_train, y_train, x_cv, y_cv, x_test_norm, y_test


def learning_curve_visualize(model, data, labels):
    train_sizes, train_scores, test_scores = learning_curve(model, data, labels, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.grid()


def create_and_deploy_logistic_model(x_train, y_train, x_cv, y_cv, org_data):
    # After tuning hyperparameters using the GridSearchCV method, the following parameters have been used.
    log_model = LogisticRegression(solver='liblinear', penalty='l2', random_state=42, C=10)
    log_model.fit(x_train, y_train)

    train_preds = log_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    cv_preds = log_model.predict(x_cv)
    cv_accuracy = accuracy_score(y_cv, cv_preds)

    print(f"Logistic Regression")
    print(f"--------------------")
    print(f"[Training] Accuracy: {train_accuracy * 100}")
    print(f"[Cross-validation] Accuracy: {cv_accuracy * 100}\n")

    # Cross-validation predictions classification report
    log_report = classification_report(y_cv, cv_preds)
    print(log_report)
    print()

    plt.figure(figsize=(10, 5))
    plt.suptitle('Logistic Regression')
    plt.subplot(1, 2, 1)
    plt.title('Learning Curve')
    learning_curve_visualize(model=log_model, data=org_data[0], labels=org_data[1])

    # Cross-validation predictions confusion matrix
    log_cm = confusion_matrix(y_cv, cv_preds)
    plt.subplot(1, 2, 2)
    sns.heatmap(log_cm, annot=True, linewidths=0.1)
    plt.title('Cross-validation predictions confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.show()

    return log_model, dict(training_acc=train_accuracy, cross_validation_acc=cv_accuracy)


def create_and_deploy_decision_tree_model(x_train, y_train, x_cv, y_cv, org_data):
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=42, max_leaf_nodes=10)
    tree_model.fit(x_train, y_train)

    train_preds = tree_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    cv_preds = tree_model.predict(x_cv)
    cv_accuracy = accuracy_score(y_cv, cv_preds)

    print(f"Decision Tree Classifier")
    print(f"--------------------")
    print(f"[Training] Accuracy: {train_accuracy * 100}")
    print(f"[Cross-validation] Accuracy: {cv_accuracy * 100}\n")

    # Cross-validation predictions classification report
    tree_report = classification_report(y_cv, cv_preds)
    print(tree_report)
    print()

    plt.figure(figsize=(10, 5))
    plt.suptitle('Decision Tree')
    plt.subplot(1, 2, 1)
    plt.title('Learning Curve')
    learning_curve_visualize(model=tree_model, data=org_data[0], labels=org_data[1])

    # Cross-validation predictions confusion matrix
    tree_cm = confusion_matrix(y_cv, cv_preds)
    plt.subplot(1, 2, 2)
    plt.title('Cross-validation predictions confusion matrix')
    sns.heatmap(tree_cm, annot=True, linewidths=0.1)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    return tree_model, dict(training_acc=train_accuracy, cross_validation_acc=cv_accuracy)


def create_and_deploy_knn_model(x_train, y_train, x_cv, y_cv, org_data):
    knn_model = KNeighborsClassifier(n_neighbors=15, p=3, weights='uniform', leaf_size=1)
    knn_model.fit(x_train, y_train)

    train_preds = knn_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    cv_preds = knn_model.predict(x_cv)
    cv_accuracy = accuracy_score(y_cv, cv_preds)

    print(f"KNN Classifier")
    print(f"--------------------")
    print(f"[Training] Accuracy: {train_accuracy * 100}")
    print(f"[Cross-validation] Accuracy: {cv_accuracy * 100}\n")

    # Cross-validation predictions classification report
    knn_report = classification_report(y_cv, cv_preds)
    print(knn_report)
    print()

    plt.figure(figsize=(10, 5))
    plt.suptitle('K-Nearest Neighbor (KNN)')
    plt.subplot(1, 2, 1)
    plt.title('Learning Curve')
    learning_curve_visualize(model=knn_model, data=org_data[0], labels=org_data[1])

    # Cross-validation predictions confusion matrix
    knn_cm = confusion_matrix(y_cv, cv_preds)
    plt.subplot(1, 2, 2)
    sns.heatmap(knn_cm, annot=True, linewidths=0.1)
    plt.title('Cross-validation predictions confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    return knn_model, dict(training_acc=train_accuracy, cross_validation_acc=cv_accuracy)


def create_and_deploy_random_forest_model(x_train, y_train, x_cv, y_cv, org_data):
    forest_model = RandomForestClassifier(max_depth=7, random_state=42)
    forest_model.fit(x_train, y_train)

    train_preds = forest_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    cv_preds = forest_model.predict(x_cv)
    cv_accuracy = accuracy_score(y_cv, cv_preds)

    print(f"Random Forest Classification")
    print(f"--------------------")
    print(f"[Training] Accuracy: {train_accuracy * 100}")
    print(f"[Cross-validation] Accuracy: {cv_accuracy * 100}\n")

    # Cross-validation predictions classification report
    forest_report = classification_report(y_cv, cv_preds)
    print(forest_report)
    print()

    plt.figure(figsize=(10, 5))
    plt.suptitle('Random Forest Classification')
    plt.subplot(1, 2, 1)
    plt.title('Learning Curve')
    learning_curve_visualize(model=forest_model, data=org_data[0], labels=org_data[1])

    # Cross-validation predictions confusion matrix
    forest_cm = confusion_matrix(y_cv, cv_preds)
    plt.subplot(1, 2, 2)
    sns.heatmap(forest_cm, annot=True, linewidths=0.1)
    plt.title('Cross-validation predictions confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    return forest_model, dict(training_acc=train_accuracy, cross_validation_acc=cv_accuracy)


def generate_confusion_matrix_testing_data(real_labels, log_preds, tree_preds, knn_preds, forest_preds):
    plt.figure(figsize=(10, 10))
    plt.suptitle('Confusion matrix - Testing data')

    plt.subplot(2, 2, 1)
    sns.heatmap(confusion_matrix(real_labels, log_preds), annot=True, linewidths=0.1)
    plt.title('Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    plt.subplot(2, 2, 2)
    sns.heatmap(confusion_matrix(real_labels, tree_preds), annot=True, linewidths=0.1)
    plt.title('Decision Tree')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    plt.subplot(2, 2, 3)
    sns.heatmap(confusion_matrix(real_labels, knn_preds), annot=True, linewidths=0.1)
    plt.title('KNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    plt.subplot(2, 2, 4)
    sns.heatmap(confusion_matrix(real_labels, forest_preds), annot=True, linewidths=0.1)
    plt.title('Random Forest Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    plt.show()


def main():
    csv_file = 'training_data.csv'
    data_df = data_imputation(csv_path=csv_file)
    X, y = preprocess_data(df=data_df)
    X_, y_, x_train, y_train, x_cv, y_cv, x_test, y_test = split_and_normalize(X=X, y=y)
    logistic_model, log_scores = create_and_deploy_logistic_model(x_train=x_train, y_train=y_train, x_cv=x_cv, y_cv=y_cv,
                                                              org_data=(X_, y_))
    decision_tree_model, tree_scores = create_and_deploy_decision_tree_model(x_train=x_train, y_train=y_train, x_cv=x_cv,
                                                                        y_cv=y_cv, org_data=(X_, y_))
    knn_model, knn_scores = create_and_deploy_knn_model(x_train=x_train, y_train=y_train, x_cv=x_cv, y_cv=y_cv,
                                                    org_data=(X_, y_))
    forest_model, forest_scores = create_and_deploy_random_forest_model(x_train=x_train, y_train=y_train, x_cv=x_cv, y_cv=y_cv,
                                                                 org_data=(X_, y_))

    # Evaluating Models on testing data which contains 6 unseen data samples from the original dataset.
    logistic_test_preds = logistic_model.predict(x_test)
    logistic_test_acc = accuracy_score(y_test, logistic_test_preds)

    tree_test_preds = decision_tree_model.predict(x_test)
    tree_test_acc = accuracy_score(y_test, tree_test_preds)

    knn_test_preds = knn_model.predict(x_test)
    knn_test_acc = accuracy_score(y_test, knn_test_preds)

    forest_test_preds = forest_model.predict(x_test)
    forest_test_acc = accuracy_score(y_test, forest_test_preds)

    generate_confusion_matrix_testing_data(real_labels=y_test, log_preds=logistic_test_preds, tree_preds=tree_test_preds,
                                           knn_preds=knn_test_preds, forest_preds=forest_test_preds)

    # Creating a pandas Dataframe for clear visualization of data
    column_titles = ['Training', 'Cross_validation', 'Testing']
    row_titles = ['Logistic Regression', 'Decision Trees', 'KNN', 'Random Forest Classifier']
    logistic_regression_scores = [log_scores['training_acc'], log_scores['cross_validation_acc'], logistic_test_acc]
    decision_tree_scores = [tree_scores['training_acc'], tree_scores['cross_validation_acc'], tree_test_acc]
    knn_all_scores = [knn_scores['training_acc'], knn_scores['cross_validation_acc'], knn_test_acc]
    forest_all_scores = [forest_scores['training_acc'], forest_scores['cross_validation_acc'], forest_test_acc]

    scores_df = pd.DataFrame(data=np.row_stack((logistic_regression_scores, decision_tree_scores, knn_all_scores, forest_all_scores)),
                             index=row_titles, columns=column_titles)

    print(scores_df)


main()
