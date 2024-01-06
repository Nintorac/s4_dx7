{% macro duckdb__create_schema(relation) -%}
  {%- call statement('create_schema') -%}
    {% set sql %}
        select type from duckdb_databases()
        where database_name='{{ relation.database }}'
        and type='sqlite'
    {% endset %}
    {% set results = run_query(sql) %}
    {% if results|length == 0 %}
        create schema if not exists {{ relation.without_identifier() }}
    {% else %}
        {% if relation.schema!='main' %}
            {{ exceptions.raise_compiler_error(
                "Schema must be 'main' when writing to sqlite "
                ~ "instead got " ~ relation.schema
            )}}
        {% endif %}
    {% endif %}
  {%- endcall -%}
{% endmacro %}