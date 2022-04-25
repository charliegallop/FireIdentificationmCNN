#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

df["group"] = df["model"].apply(lambda x: string if x.contains(string, regex = False, string))
# %%
