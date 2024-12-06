import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to the database
conn = sqlite3.connect('study.db')

# Query trials table
trials_df = pd.read_sql_query("""
    SELECT 
        t.trial_id,
        v.value,
        t.datetime_start,
        t.datetime_complete,
        p.param_name,
        p.param_value
    FROM trials t
    LEFT JOIN trial_values v ON t.trial_id = v.trial_id
    LEFT JOIN trial_params p ON t.trial_id = p.trial_id
    WHERE t.state = 'COMPLETE'
""", conn)

# Pivot the parameters into columns
trials_pivot = trials_df.pivot(
    index='trial_id',
    columns='param_name',
    values='param_value'
).reset_index()

# Add the objective value
trials_pivot = trials_pivot.merge(
    trials_df[['trial_id', 'value']].drop_duplicates(),
    on='trial_id'
)

# Convert columns to numeric
for col in trials_pivot.columns:
    if col != 'trial_id':
        trials_pivot[col] = pd.to_numeric(trials_pivot[col], errors='ignore')

# Print basic statistics
print("\n=== Trial Statistics ===")
print(f"Total completed trials: {len(trials_pivot)}")
print("\nBest trial:")
best_trial = trials_pivot.loc[trials_pivot['value'].idxmax()]
print(f"Trial ID: {best_trial['trial_id']}")
print(f"Objective value: {best_trial['value']:.4f}")
print("\nBest hyperparameters:")
for col in trials_pivot.columns:
    if col not in ['trial_id', 'value']:
        print(f"{col}: {best_trial[col]}")

# Plot parameter importance
plt.figure(figsize=(10, 6))
correlations = trials_pivot.corr()['value'].sort_values(ascending=False)
correlations = correlations.drop('value')
correlations.plot(kind='bar')
plt.title('Parameter Importance (Correlation with Objective)')
plt.tight_layout()
plt.savefig('../docs/parameter_importance.png')

# Plot objective value history
plt.figure(figsize=(10, 6))
plt.plot(trials_pivot['trial_id'], trials_pivot['value'])
plt.title('Optimization History')
plt.xlabel('Trial ID')
plt.ylabel('Objective Value')
plt.tight_layout()
plt.savefig('../docs/optimization_history.png')

# Close connection
conn.close()
