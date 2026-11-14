"""Optional, deliberately decoupled MySQL persistence helper."""

import os


def save_to_mysql(dataframe, table_name, if_exists="append"):
    """Write a DataFrame using LOYALTYSYNTH_DATABASE_URL when configured.

    MySQL is never contacted just by starting the web app. Table names are
    restricted to simple SQL identifiers before passing through pandas.
    """
    import re
    from sqlalchemy import create_engine

    url = os.environ.get("LOYALTYSYNTH_DATABASE_URL")
    if not url:
        raise RuntimeError("Set LOYALTYSYNTH_DATABASE_URL before using the MySQL adapter")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", table_name):
        raise ValueError("table_name must be a simple identifier (letters, numbers, underscores)")
    if if_exists not in {"fail", "replace", "append"}:
        raise ValueError("if_exists must be fail, replace, or append")
    engine = create_engine(url)
    try:
        dataframe.to_sql(table_name, engine, if_exists=if_exists, index=False)
    finally:
        engine.dispose()
