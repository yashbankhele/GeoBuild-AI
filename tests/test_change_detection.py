import numpy as np
import pytest
from src.change_detection import compare_images
from src.evaluation import evaluate_binary_mask


def test_identical_images_have_no_change():
    img = np.zeros((32,32,3), dtype=np.uint8)
    r = compare_images(img, img.copy(), threshold=25, blur_kernel=1)
    assert r.changed_pixels == 0
    assert r.changed_percentage == 0


def test_obvious_change_is_detected():
    a = np.zeros((40,40,3), dtype=np.uint8)
    b = a.copy()
    b[10:30,10:30] = 255
    assert compare_images(a,b,threshold=20,blur_kernel=1).changed_pixels > 0


def test_invalid_threshold_raises():
    a = np.zeros((10,10,3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compare_images(a,a,threshold=300)


def test_evaluation_perfect_mask():
    truth = np.zeros((10,10), dtype=np.uint8)
    truth[2:5,2:5] = 255
    metrics = evaluate_binary_mask(truth, truth.copy())
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0


def test_evaluation_requires_matching_shape():
    with pytest.raises(ValueError):
        evaluate_binary_mask(np.zeros((4,4)), np.zeros((3,3)))


def test_eval_confusion_matrix_shape():
    truth = np.array([[0,255],[0,255]], dtype=np.uint8)
    pred = np.array([[0,255],[255,0]], dtype=np.uint8)
    metrics = evaluate_binary_mask(pred, truth)
    assert len(metrics["confusion_matrix"]) == 2
