from tqdm import tqdm


def maybe_tqdm(use_tqdm: bool, arr, **args):
    """Helper function to determine whether to use tqdm to show progress.

    Parameters
    ----------
    use_tqdm: bool
        Whether to use tqdm.
    arr: ArrayLike
        The array to iterate.
    **args:
        Other parameters passed to tqdm.

    Returns
    -------
    ArrayLike | tqdm
        ArrayLike if use_tqdm is False, otherwise tqdm.
    """
    if use_tqdm:
        return tqdm(arr, **args)
    else:
        return arr
