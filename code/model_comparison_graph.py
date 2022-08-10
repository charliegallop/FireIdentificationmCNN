#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

#%%

df = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM602_Trends_in_data_science_and_AI/Project/data/model_comparison.csv")
# %%
df.head()

#%%

df.columns

#%%
df.rename(columns={'Model':'model',
'Size (MB)': 'size', 
'Top-1 Accuracy (%)':'top1', 
'Top-5 Accuracy (%)':'top5', 
'Parameters (M)':'parameters', 
'Depth':'depth', 
'Time (ms) per inference step (CPU)':'time_CPU', 
 'Time (ms) per inference step (GPU)':'time_GPU'}, inplace=True)

#%%
df.head()
# %%

graph = alt.Chart(df).mark_point(filled = True,
        size = 100,
).encode(
    alt.X('parameters',
        scale=alt.Scale(domain=(0, 150)),
        axis=alt.Axis(title='Number of Parameters (Millions)')
    ),
    alt.Y('top1',
        scale=alt.Scale(domain=(0.68, 0.85)),
        axis=alt.Axis(format = '%', title='Top 1 Accuracy')
    ),
    color = alt.Color('group', legend=alt.Legend(title="Model Family")),
    size = alt.Size('size', legend=alt.Legend(title="Size in MB", orient = 'right'), scale = alt.Scale(range= [100, 2500]))
).properties(
    height = 500,
    width = 1000
)

text = alt.Chart(df).mark_text(
    align='left',
    baseline='top',
    dx=8,
    dy = 5,
    size = 15
).encode(
    x='parameters:Q',
    y='top1:Q',
    text=alt.Text('model')
)

chart = (graph+text).configure_axis(
    grid=False,
    labelFontSize=15,
    titleFontSize=15
).configure_view(
    strokeWidth=0
)
chart
#%%

chart.save('../plots/model_comparison.svg')

# %%
x = df["parameters"]
y = df["top1"]

plt.figure(figsize=(8,5))
sns.scatterplot(x = x, y = y, hue = 'group', data = df)
for i in range(df.shape[0]):
    plt.annotate(xy = [df.parameters[i] + 0.3, df.top1[i] + 0.3], text = df.model[i])


#%%

fig = px.scatter(df, x = 'parameters', y = 'top1', text = 'model')
fig.show()


# %%
df.dtypes

# %%
df["parameters"]
# %%

string = ResNet

df["group"] = df["model"].apply(lambda x: string if x.contains(string, regex = False))
# %%

# %%
