from functools import lru_cache
import os
import duckdb

from s4_dx7.udf import Plugin

# @lru_cache
def get_duckdb_connection(path=':memory:'):
    """
    inits the duckdb connection according to the dbt setup

    TODO: this is a total hack make this use 
    the actual dbt setup pathway
    """
    conn = duckdb.connect(path)
    Plugin.configure_connection(None, conn)
    conn.execute(f"attach '{os.environ['AUDIO_DB']}' as audio")
    return conn