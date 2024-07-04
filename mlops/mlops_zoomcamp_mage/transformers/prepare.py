from utils_fn.data_preparation.cleaning import *
from utils_fn.data_preparation.preprocessing import prep
from utils_fn.data_preparation.splitting import *
from typing import Tuple


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, 
        pd.Series, pd.Series
        ]:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    df = clean(df)
    return prep(df)
    # X_train, X_test, y_train, y_test = split(df)

    # return X_train, X_test, y_train, y_test


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'