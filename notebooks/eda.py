# %% [markdown]
# # Exploratory Data Analysis of Phrase Stats
# 
# In this notebook, we will perform an exploratory data analysis on the `phrase_stats` table from the DuckDB database located at `data/dev.duckdb`. We will use Plotly Express for visualization.

# %% [markdown]
# ## Setup and Data Loading
# First, we need to import the necessary libraries and load the data from the DuckDB database.

# %%
# Importing necessary libraries
import os
import duckdb
import pandas as pd
import plotly.express as px

# %%
# Connect to the DuckDB database
conn = duckdb.connect(os.environ['WORKING_DB'])

# %%
# Load the data into a DataFrame
query = "SELECT * FROM dev.main.phrase_stats USING SAMPLE 10% (bernoulli);"
df_phrase_stats = conn.execute(query).fetchdf()

# %%
# Display the first few rows of the DataFrame
df_phrase_stats.head()

# %% [markdown]
# ## Data Overview
# Let's get a quick overview of the data types and the presence of null values in our dataset.

# %%
# Data types and null values
df_phrase_stats.info()
# %% [markdown]
# ## Visualizations
# Now, let's create some visualizations to understand the data better.

# %% [markdown]
# ### Distribution of Sum Duration
# We can start by looking at the distribution of the `sum_duration` column.

# %%
df_phrase_stats = df_phrase_stats[(df_phrase_stats['sum_duration']<40)]
df_phrase_stats = df_phrase_stats[(df_phrase_stats['n_notes']<34)]
df_phrase_stats = df_phrase_stats[(df_phrase_stats['polyphony']<10)]

fig_sum_duration = px.histogram(df_phrase_stats, x='sum_duration', nbins=50, title='Distribution of Sum Duration')
fig_sum_duration.show()

# %% [markdown]
# ### Number of Notes vs. Sum Duration
# It might be interesting to see if there's a relationship between the number of notes and the sum duration.

# %%
fig_notes_duration = px.scatter(df_phrase_stats, x='n_notes', y='sum_duration', title='Number of Notes vs. Sum Duration')
fig_notes_duration.show()

# %% [markdown]
# ### Polyphony Distribution
# Let's visualize the distribution of polyphony values.

# %%
fig_polyphony = px.histogram(df_phrase_stats, x='polyphony', title='Polyphony Distribution')
fig_polyphony.show()

# %% [markdown]
# ### Min and Max Note Distribution
# We can also look at the distribution of the minimum and maximum note values.

# %%
fig_min_note = px.histogram(df_phrase_stats, x='min_note', nbins=30, title='Min Note Distribution')
fig_min_note.show()

fig_max_note = px.histogram(df_phrase_stats, x='max_note', nbins=30, title='Max Note Distribution')
fig_max_note.show()

# %% [markdown]
# ### Polyphony vs. Sum Duration
# Let's see if there's a trend between polyphony and sum duration.

# %%
fig_polyphony_duration = px.scatter(df_phrase_stats, x='polyphony', y='sum_duration', title='Polyphony vs. Sum Duration')
fig_polyphony_duration.show()

# %% [markdown]
# ### Correlation Heatmap
# Finally, let's create a heatmap to visualize the correlation between all numerical features.

# %%
df_corr = df_phrase_stats[['sum_duration', 'n_notes', 'min_note', 'max_note', 'polyphony',]].corr()
fig_corr = px.imshow(df_corr, aspect="auto", title='Correlation Heatmap')
fig_corr.show()

#%%
# %% [markdown]
# ### Box Plots for Statistical Distribution

# %%
# Box plot for sum_duration
fig_box_sum_duration = px.box(df_phrase_stats, y='sum_duration', title='Box Plot of Sum Duration')
fig_box_sum_duration.show()

# Box plot for n_notes
fig_box_n_notes = px.box(df_phrase_stats, y='n_notes', title='Box Plot of Number of Notes')
fig_box_n_notes.show()

# %% [markdown]
# ### Phrase Complexity Analysis

# %%
# Scatter plot for phrase complexity (n_notes vs polyphony)
fig_complexity = px.scatter(df_phrase_stats, x='n_notes', y='polyphony', color='sum_duration',
                            title='Phrase Complexity: Number of Notes vs Polyphony')
fig_complexity.show()

# %% [markdown]
# ### Polyphony and Note Range Relationship

# %%
# Scatter plot for polyphony vs note range
df_phrase_stats['note_range'] = df_phrase_stats['max_note'] - df_phrase_stats['min_note']
fig_polyphony_range = px.scatter(df_phrase_stats, x='polyphony', y='note_range', title='Polyphony vs Note Range')
fig_polyphony_range.show()

# %% [markdown]
# ### Heatmap of Note Range vs. Polyphony

# %%
# Heatmap (or hexbin plot) of note range vs polyphony
fig_heatmap_range_polyphony = px.density_heatmap(df_phrase_stats, x='polyphony', y='note_range', nbinsx=20, nbinsy=20,
                                                 title='Heatmap of Note Range vs Polyphony')
fig_heatmap_range_polyphony.show()

# %% [markdown]
# ### Cumulative Distribution Function (CDF)

# %%
# CDF for sum_duration
fig_cdf_sum_duration = px.ecdf(df_phrase_stats, x='sum_duration', title='CDF of Sum Duration')
fig_cdf_sum_duration.show()

# CDF for n_notes
fig_cdf_n_notes = px.ecdf(df_phrase_stats, x='n_notes', title='CDF of Number of Notes')
fig_cdf_n_notes.show()
# %% [markdown]
# ## Conclusion
# This exploratory analysis provides an initial understanding of the phrase stats data. Further analysis could include more complex visualizations, statistical tests, and machine learning models to uncover deeper insights.