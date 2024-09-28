import asyncio
from duckdb import DuckDBPyConnection
from dbt.adapters.duckdb.plugins import BasePlugin
from s4_dx7.udf.digital import fsk_encode_syx_udf

from s4_dx7.udf.dx7 import clean_voice_udf, extract_voices_udf
from s4_dx7.udf.render import render_dx7_udf, render_midi_udf

class Plugin(BasePlugin):
    def configure_connection(self, conn: DuckDBPyConnection):
        conn.create_function("render_midi", render_midi_udf, type="arrow")
        conn.create_function("extract_voices", extract_voices_udf, type="arrow")
        conn.create_function("clean_voice", clean_voice_udf, type="arrow")
        conn.create_function("fsk_encode_syx", fsk_encode_syx_udf, type="arrow")
        conn.create_function("render_dx7", render_dx7_udf, type="arrow")
        