from dataload import Record, load_records

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from transform import get_hodge_spectrum_dataset
from metrics import metrics_report


def train_hodge_specter():
    records = load_records('data/minirecords.pkl')
    records = [r for r in records if r.atlas_name == 'BASC']

    X, y = get_hodge_spectrum_dataset(records, [0, 1, 2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    clf = CatBoostClassifier(verbose=0, task_type='GPU')
    clf.fit(X_train, y_train)

    print(metrics_report(clf, X_test, y_test))


def main():
    train_hodge_specter()


if __name__ == '__main__':
    main()
