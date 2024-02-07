import asyncio
from duckdb import DuckDBPyConnection
from dbt.adapters.duckdb.plugins import BasePlugin

from s4_dx7.udf.dx7 import extract_voices_udf
from s4_dx7.udf.render import render_midi_udf

class Plugin(BasePlugin):
    def configure_connection(self, conn: DuckDBPyConnection):
        conn.create_function("render_midi", render_midi_udf, type="arrow")
        conn.create_function("extract_voices", extract_voices_udf, type="arrow")
        