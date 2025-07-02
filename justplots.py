import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from prePro import processed_data
df=processed_data
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
under_2m_count = df[df['price'] < 2_000_000].shape[0]
print(f"Houses under 2M: {under_2m_count}")

plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=40, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of House Prices', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Count', fontsize=12)

formatter = ticker.FuncFormatter(lambda x, _: f'{int(x / 1e6)}M')
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(0, df['price'].max())
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['area'], bins=40, kde=True, color='lightgreen', edgecolor='black')

plt.title('Distribution of House Area', fontsize=16)
plt.xlabel('Area (sq ft)', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

df_plot = df.copy()
df_plot['furnishingstatus'] = df_plot['furnishingstatus'].map({
    0: 'Unfurnished',
   -1: 'Semi-furnished',
    1: 'Furnished'
})

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_plot, x='furnishingstatus', y='price', palette='Set2')

plt.title('House Prices by Furnishing Status', fontsize=16)
plt.xlabel('Furnishing Status', fontsize=12)
plt.ylabel('Price', fontsize=12)

# Format y-axis to millions
formatter = ticker.FuncFormatter(lambda x, _: f'{int(x / 1e6)}M')
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(df['area'], df['price'], c='blue', edgecolors='black', alpha=0.7)
plt.title('Scatterplot of House Price vs Area', fontsize=14)
plt.xlabel('Area (sq ft)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True)
plt.show()
