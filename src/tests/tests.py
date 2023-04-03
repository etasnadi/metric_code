import os

import numpy as np

from stardist.matching import matching_dataset

stardist_verbose = False

eps = .0001

def eqf(a, b):
    return (a-b)**2 < eps

def test_stardist_matching_single_image():
    print("%s\tTest on a single image...%s" % (os.linesep, os.linesep))
    # ----------------------------------------------------------------------------------------------
    # Testing with gold and predicted masks
    # ----------------------------------------------------------------------------------------------

    mask_1 = np.zeros((128, 128), dtype=np.uint16)
    mask_2 = np.zeros((128, 128), dtype=np.uint16)

    # Draw object #1: 70% overlap
    mask_1[20:30, 20:30] = 1
    mask_2[20:30, 23:30] = 1

    # Draw object #2: 90% overlap
    mask_1[30:40, 20:30] = 2
    mask_2[30:40, 21:30] = 2

    # Draw object #3: 0% overlap
    mask_1[40:50, 20:30] = 3
    mask_2[40:50, 30:40] = 3

    # Draw object #4 (only on the first image)
    mask_1[40:50, 90:100] = 4

    # Assumed score matrix:
    # [[0.7 0.  0. ]
    #  [0.  0.9 0. ]
    #  [0.  0.  0. ]
    #  [0.  0.  0. ]]
    assumed_scores = np.array([
        [0.7,   0.,     0.  ], 
        [0.,    0.9,    0.  ], 
        [0.,    0.,     0.  ], 
        [0.,    0.,     0.  ]])

    matching = matching_dataset([mask_1], [mask_2], scores_input=False)
    
    # Assumed object counts
    a_tp = 2
    a_fn = 2
    a_fp = 1
    
    # Resulting object counts
    tp = matching.tp
    fn = matching.fn
    fp = matching.fp
    
    assert tp==a_tp, ValueError("Test failed (tp)")
    assert fn==a_fn, ValueError("Test failed (fn)")
    assert fp==a_fp, ValueError("Test failed (fp)")

    assumed_digits_score = tp**2/(tp**2+tp*fn+tp*fp+fp*fn)
    assert eqf(assumed_digits_score, matching.digits_score), ValueError("Test failed (digits score)")

    # ----------------------------------------------------------------------------------------------
    # Testing with direct scores
    # ----------------------------------------------------------------------------------------------

    '''
    If scores_input is True, then the first parameter is the score matrix while the second parameter is
    the custom matching. This way, we skip the internal matching algorithm that maximizes the global iou score.
    Furthermore, we also have to define the score matrix directly as the original matrix construction algorithm 
    assumes that the objects do not overlap with each other.
    '''

    matching_objects = [
        [0, 1, 2], 
        [0, 1, 2]]  # We do not care about the fourth ground truth object as it does not overlap with any 
    # predicted object.

    matching = matching_dataset(
        [assumed_scores],   # Score matrix
        [matching_objects], # Matched object ids
        scores_input=True, verbose=stardist_verbose)
    
    # The result should be the same
    assert matching.tp==a_tp, ValueError("Test failed (tp#2)")
    assert matching.fn==a_fn, ValueError("Test failed (fn#2)")
    assert matching.fp==a_fp, ValueError("Test failed (fp#2)")

def test_stardist_matching_multiple_images_1():
    print("%s\tTest on multiple images with by_image=True%s" % (os.linesep, os.linesep))
    # ----------------------------------------------------------------------------------------------
    # Testing with multiple gold and predicted masks
    # ----------------------------------------------------------------------------------------------

    # Image pair "a"

    mask_1a = np.zeros((128, 128), dtype=np.uint16)
    mask_2a = np.zeros((128, 128), dtype=np.uint16)

    # Draw object #1: 70% overlap
    mask_1a[20:30, 20:30] = 1
    mask_2a[20:30, 23:30] = 1

    # Draw object #2: 90% overlap
    mask_1a[30:40, 20:30] = 2
    mask_2a[30:40, 21:30] = 2

    # Draw object #3: 0% overlap
    mask_1a[40:50, 20:30] = 3
    mask_2a[40:50, 30:40] = 3

    # Draw object #4 (only on the first image)
    mask_1a[40:50, 90:100] = 4

    # Image pair "b"

    assumed_scores_a = np.array([
        [0.7,   0.,     0.  ], 
        [0.,    0.9,    0.  ], 
        [0.,    0.,     0.  ], 
        [0.,    0.,     0.  ]])

    mask_1b = np.zeros((256, 256), dtype=np.uint16)
    mask_2b = np.zeros((256, 256), dtype=np.uint16)

    # Draw object #1: 60% overlap
    mask_1b[20:30, 20:30] = 1
    mask_2b[20:30, 24:30] = 1

    # Draw object #2: 80% overlap
    mask_1b[230:240, 220:230] = 2
    mask_2b[230:240, 222:230] = 2

    # Draw object #3: 0% overlap
    mask_1b[40:50, 20:30] = 3
    mask_2b[40:50, 30:40] = 3

    # Draw object #4 (only on the second image)
    mask_2b[40:50, 90:100] = 4

    # Draw object #5 (only on the second image)
    mask_2b[50:60, 90:100] = 5

    assumed_scores_b = np.array([
        [0.6,   0.,     0.,     0.,     0.], 
        [0.,    0.8,    0.,     0.,     0.], 
        [0.,    0.,     0.,     0.,     0.]])

    matching = matching_dataset([mask_1a, mask_1b], [mask_2a, mask_2b], scores_input=False, by_image=True)
    
    # Assumed object counts
    a_tp_a = 2
    a_fn_a = 2
    a_fp_a = 1
    
    a_tp_b = 2
    a_fn_b = 1
    a_fp_b = 3

    # Resulting object counts
    tp = matching.tp
    fn = matching.fn
    fp = matching.fp
    
    assert tp==a_tp_a+a_tp_b, ValueError("Test failed (tp)")
    assert fn==a_fn_a+a_fn_b, ValueError("Test failed (fn)")
    assert fp==a_fp_a+a_fp_b, ValueError("Test failed (fp)")

    assumed_digits_score_a = a_tp_a**2/(a_tp_a**2+a_tp_a*a_fn_a+a_tp_a*a_fp_a+a_fp_a*a_fn_a)
    assumed_digits_score_b = a_tp_b**2/(a_tp_b**2+a_tp_b*a_fn_b+a_tp_b*a_fp_b+a_fp_b*a_fn_b)
    assumed_digits_score = .5*(assumed_digits_score_a + assumed_digits_score_b)
    assert eqf(assumed_digits_score, matching.digits_score), ValueError("Test failed (digits score)")

    # ----------------------------------------------------------------------------------------------
    # Testing with direct scores
    # ----------------------------------------------------------------------------------------------

    '''
    If scores_input is True, then the first parameter is the score matrix while the second parameter is
    the custom matching. This way, we skip the internal matching algorithm that maximizes the global iou score.
    Furthermore, we also have to define the score matrix directly as the original matrix construction algorithm 
    assumes that the objects do not overlap with each other.
    '''

    matching_objects_a = [
        [0, 1, 2], 
        [0, 1, 2]]

    matching_objects_b = [
        [0, 1, 2], 
        [0, 1, 2]]

    matching = matching_dataset(
        [assumed_scores_a, assumed_scores_b],       # Score matrix
        [matching_objects_a, matching_objects_b],   # Matched object ids
        scores_input=True, verbose=stardist_verbose, by_image=True)
    
    # The result should be the same
    assert matching.tp==a_tp_a+a_tp_b, ValueError("Test failed (tp#2)")
    assert matching.fn==a_fn_a+a_fn_b, ValueError("Test failed (fn#2)")
    assert matching.fp==a_fp_a+a_fp_b, ValueError("Test failed (fp#2)")
    assert eqf(matching.digits_score, assumed_digits_score), ValueError("Test failed (digits score#2)")

def test_stardist_matching_multiple_images_2():
    print("%s\tTest on multiple images with by_image=False...%s" % (os.linesep, os.linesep))
    # ----------------------------------------------------------------------------------------------
    # Testing with multiple gold and predicted masks
    # ----------------------------------------------------------------------------------------------

    # Image pair "a"

    mask_1a = np.zeros((128, 128), dtype=np.uint16)
    mask_2a = np.zeros((128, 128), dtype=np.uint16)

    # Draw object #1: 70% overlap
    mask_1a[20:30, 20:30] = 1
    mask_2a[20:30, 23:30] = 1

    # Draw object #2: 90% overlap
    mask_1a[30:40, 20:30] = 2
    mask_2a[30:40, 21:30] = 2

    # Draw object #3: 0% overlap
    mask_1a[40:50, 20:30] = 3
    mask_2a[40:50, 30:40] = 3

    # Draw object #4 (only on the first image)
    mask_1a[40:50, 90:100] = 4

    # Image pair "b"

    assumed_scores_a = np.array([
        [0.7,   0.,     0.  ], 
        [0.,    0.9,    0.  ], 
        [0.,    0.,     0.  ], 
        [0.,    0.,     0.  ]])

    mask_1b = np.zeros((256, 256), dtype=np.uint16)
    mask_2b = np.zeros((256, 256), dtype=np.uint16)

    # Draw object #1: 60% overlap
    mask_1b[20:30, 20:30] = 1
    mask_2b[20:30, 24:30] = 1

    # Draw object #2: 80% overlap
    mask_1b[230:240, 220:230] = 2
    mask_2b[230:240, 222:230] = 2

    # Draw object #3: 0% overlap
    mask_1b[40:50, 20:30] = 3
    mask_2b[40:50, 30:40] = 3

    # Draw object #4 (only on the second image)
    mask_2b[40:50, 90:100] = 4

    # Draw object #5 (only on the second image)
    mask_2b[50:60, 90:100] = 5

    assumed_scores_b = np.array([
        [0.6,   0.,     0.,     0.,     0.], 
        [0.,    0.8,    0.,     0.,     0.], 
        [0.,    0.,     0.,     0.,     0.]])

    matching = matching_dataset([mask_1a, mask_1b], [mask_2a, mask_2b], scores_input=False, by_image=False)
    
    # Assumed object counts
    a_tp_a = 2
    a_fn_a = 2
    a_fp_a = 1
    
    a_tp_b = 2
    a_fn_b = 1
    a_fp_b = 3

    # Resulting object counts
    tp = matching.tp
    fn = matching.fn
    fp = matching.fp
    
    assert tp==a_tp_a+a_tp_b, ValueError("Test failed (tp)")
    assert fn==a_fn_a+a_fn_b, ValueError("Test failed (fn)")
    assert fp==a_fp_a+a_fp_b, ValueError("Test failed (fp)")

    a_sum_tp = a_tp_a + a_tp_b
    a_sum_fn = a_fn_a + a_fn_b
    a_sum_fp = a_fp_a + a_fp_b

    assumed_digits_score = a_sum_tp**2/(a_sum_tp**2+a_sum_tp*a_sum_fn+a_sum_tp*a_sum_fp+a_sum_fp*a_sum_fn)
    assert eqf(assumed_digits_score, matching.digits_score), ValueError("Test failed (digits score)")

    # ----------------------------------------------------------------------------------------------
    # Testing with direct scores
    # ----------------------------------------------------------------------------------------------

    '''
    If scores_input is True, then the first parameter is the score matrix while the second parameter is
    the custom matching. This way, we skip the internal matching algorithm that maximizes the global iou score.
    Furthermore, we also have to define the score matrix directly as the original matrix construction algorithm 
    assumes that the objects do not overlap with each other.
    '''

    matching_objects_a = [
        [0, 1, 2], 
        [0, 1, 2]]

    matching_objects_b = [
        [0, 1, 2], 
        [0, 1, 2]]

    matching = matching_dataset(
        [assumed_scores_a, assumed_scores_b],       # Score matrix
        [matching_objects_a, matching_objects_b],   # Matched object ids
        scores_input=True, verbose=stardist_verbose, by_image=False)
    
    # The result should be the same
    assert matching.tp==a_tp_a+a_tp_b, ValueError("Test failed (tp#2)")
    assert matching.fn==a_fn_a+a_fn_b, ValueError("Test failed (fn#2)")
    assert matching.fp==a_fp_a+a_fp_b, ValueError("Test failed (fp#2)")
    assert matching.digits_score-assumed_digits_score < eps, ValueError("Test failed (digits score#2)")

def run_tests():
    n_failed = 0
    n_passed = 0
    tests = [
        test_stardist_matching_single_image,
        test_stardist_matching_multiple_images_1,
        test_stardist_matching_multiple_images_2
    ]

    for t in tests:
        try:
            t()
            n_passed += 1
        except:
            n_failed += 1

    print('%s\tSummary: tests: %d passed: %d failed: %d.%s' % (os.linesep, len(tests), n_passed, n_failed, os.linesep))

if __name__ == '__main__':
    run_tests()