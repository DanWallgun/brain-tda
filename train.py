from dataload import Record, load_records

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

import transform
from metrics import metrics_report


def train_hodge_specter():
    records = load_records('data/minirecords.pkl')
    records = [r for r in records if r.atlas_name == 'BASC']

    X, y = transform.get_hodge_spectrum_dataset(records, [0, 1, 2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    clf = CatBoostClassifier(verbose=0)
    clf.fit(X_train, y_train)

    print(metrics_report(clf, X_test, y_test))


def train_betti_curves():
    records = load_records('data/minirecords.pkl')
    records = [r for r in records if r.atlas_name == 'BASC']

    X, y = transform.get_betti_curves_dataset(records)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    clf = CatBoostClassifier(verbose=0)
    clf.fit(X_train, y_train)

    print(metrics_report(clf, X_test, y_test))


def train_persistence_images():
    records = load_records('data/minirecords.pkl')
    records = [r for r in records if r.atlas_name == 'BASC']

    X, y = transform.get_persistence_images_dataset(records)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    clf = CatBoostClassifier(verbose=1, iterations=200)
    clf.fit(X_train, y_train)

    print(metrics_report(clf, X_test, y_test))


def main():
    # train_hodge_specter()
    # train_betti_curves()
    train_persistence_images()


if __name__ == '__main__':
    main()
