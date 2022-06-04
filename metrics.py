from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def metrics_report(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    return {
        'accuracy_score': accuracy_score(y_test, y_pred),
        'roc_auc_score': roc_auc_score(y_test, y_score),
        'f1_score': f1_score(y_test, y_pred),
        'balanced_accuracy_score': balanced_accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
