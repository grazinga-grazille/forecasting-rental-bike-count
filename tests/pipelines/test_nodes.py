import pandas as pd
from forecasting_rental_bike_count.pipelines.nodes import rename_columns


def test_rename_columns():
    df = pd.DataFrame({"old_name": [1, 2]})
    result = rename_columns(df, {"old_name": "new_name"})
    assert "new_name" in result.columns
    assert "old_name" not in result.columns