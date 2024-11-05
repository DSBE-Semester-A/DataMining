def highlight_best_worst(s):
    """
    Highlight the max and min values in the DataFrame or Series.
    Assume s is a column named 'mean_test_score'.
    """
    is_max = s == s.max()
    is_min = s == s.min()
    return ['background-color: limegreen' if v else 'background-color: salmon' if is_min.iloc[i] else '' for i, v in enumerate(is_max)]
