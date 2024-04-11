import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
overweight = lambda x: 1 if x > 25 else 0 
df['overweight'] = df.apply(lambda row: overweight(row['weight'] / ((row['height'] / 100) ** 2)), axis=1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
for item in df.columns:
    if item == 'cholesterol' or item == 'gluc':
    # Apply the normalization
        normalize_zero = df[item] == 1
        normalize_one = df[item] > 1
        df.loc[normalize_zero, item] = 0
        df.loc[normalize_one, item] = 1

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(
        df,  
        id_vars=["id", "cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_group = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # Draw the catplot with 'sns.catplot()'
    plot = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_group, height=5, aspect=1.5)

    # Get the figure for the output
    fig = plt.gcf()


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 11))

    # Draw the heatmap with 'sns.heatmap()'

    plt.xticks(rotation=90) 

    # Create the heatmap with Seaborn
    heatmap = sns.heatmap(
        corr, mask=mask,
        annot=True,
        square=True, fmt=".1f",
        ax=ax  # Specify the axis to plot on
    )

    ax.set_yticklabels(labels=df_heat.columns, rotation=0)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
