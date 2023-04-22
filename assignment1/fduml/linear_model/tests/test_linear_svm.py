"""
linear svm classifier test
"""
import numpy as np
from numpy.testing import assert_array_equal
from fduml.linear_model import LinearSVMClassifier


def test_linear_svm():
    """Check that the model is able to fit the classification data"""
    X = np.array([[-1, 0], [0, 1], [1, 1]])
    y = np.array([2, 1, 0])
    n_samples = len(y)

    clf = LinearSVMClassifier(loss_type='naive')
    clf.fit(X, y)
    predicted = clf.predict(X)

    assert predicted.shape == (n_samples,)
    assert_array_equal(predicted, y)

    clf = LinearSVMClassifier(loss_type='vectorized')
    clf.fit(X, y)
    predicted = clf.predict(X)

    assert predicted.shape == (n_samples,)
    assert_array_equal(predicted, y)
    print("test_linear_svm passed")


test_linear_svm()
