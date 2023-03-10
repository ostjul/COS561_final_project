import numpy as np
from sklearn import cluster

def get_bin(prediction, bins):
        '''
        Return index of bin that error falls into
        '''
        n_bins = bins.shape[0]
        # Check for prediction out of bounds at low end
        if prediction <= bins[0]:
            return 0
        # Check for prediction out of bounds at high end
        elif prediction > bins[-1]:
            return n_bins - 1
        # Find bin prediction falls in
        bin_idxs = np.where(prediction <= bins)[0]
        return bin_idxs[0] - 1

def create_bins(predictions, targets, n_bins):
    '''
    Given the prediction sojourn times and target sojourn times,
        1) calculate errors
        2) linearly bin predictions
        3) calculate the average error for each bin
        4) return bins and average errors

    Arg(s):
        predictions : 1D np.array
            predicted sojourn times from training data
        targets : 1D np.array
            true sojourn times from training data
        n_bins : int
            number of bins

    Returns:
        bins : 1D np.array of length n_bins + 1
        average_errors : 1D np.array of length n_bins
    '''
    assert predictions.shape == targets.shape
    assert len(predictions.shape) == 1

    # Calculate errors
    error = predictions - targets

    # Create bins
    min_prediction = np.amin(predictions)
    max_prediction = np.amax(predictions)
    bins = np.linspace(min_prediction, max_prediction, num=n_bins+1)

    # Obtain which bin each prediction is in
    bin_idxs = predictions.apply(lambda x: get_bin(x, bins))

    # Calculate average error per bin
    average_errors = []
    for i in range(n_bins):
        idxs = np.where(bin_idxs == i)
        bin_errors = errors[idxs]
        average_errors.append(np.mean(bin_errors))

    average_errors = np.array(average_errors)

    assert len(bins.shape[0]) == len(average_errors.shape) + 1

    return bins, average_errors

def correct_predictions(predictions, bins, average_errors):
    '''
    Given predictions, bins, and average_errors, subtract the average error of that bin from the prediction

    Arg(s):
        predictions : 1D np.array
            current sojourn time predictions
        bins : 1D np.array
            length n_bins + 1, marks the edges of each bin
        average_errors : 1D np.array
            average error for each bin
    '''
    # Obtain which bin each prediction is in
    bin_idxs = predictions.apply(lambda x: get_bin(x, bins))

    # Calculate how much error to subtract from each prediction based on bin index
    corresponding_errors = bin_idxs.apply(lambda idx: average_errors[idx])

    # Adjust prediction
    predictions = predictions - corresponding_errors
    return predictions