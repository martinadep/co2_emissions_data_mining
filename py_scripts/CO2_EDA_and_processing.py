"""
Raw CO2 Emissions Dataset - Processing Script
This script performs EDA and processing on raw CO2 emissions datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use('ggplot')
pd.set_option('display.max_columns', 100)

print("="*80)
print("RAW CO2 EMISSIONS - DATA PROCESSING")
print("="*80)

# ============================================================================
# Step 0: Import and Read Data
# ============================================================================
print("\n[Step 0] Loading raw data...")

df_co2_emissions_per_capita = pd.read_csv('../data/raw/co2_emissions/co2-emissions-per-capita.csv')
df_co2_emissions = pd.read_csv('../data/raw/co2_emissions/co2-emissions-by-region.csv')

print(">>> Data loaded successfully")

# ============================================================================
# Step 1: Data Understanding
# ============================================================================
print("-"*80)
print("\n[Step 1] Data Understanding")

# 1.1 CO2 emissions per capita
print("\n1.1 CO2 Emissions Per Capita:")
print(f"Shape: {df_co2_emissions_per_capita.shape}")
print(f"Missing values:\n{df_co2_emissions_per_capita.isna().sum()}")
print(f"\nEntities without Code (geographical regions):")
print(df_co2_emissions_per_capita[df_co2_emissions_per_capita['Code'].isna()]['Entity'].unique())
print("-"*80)

# 1.2 CO2 emissions
print("\n1.2 CO2 emissions:")
print(f"Shape: {df_co2_emissions.shape}")
print(f"Missing values:\n{df_co2_emissions.isna().sum()}")
print(f"\nEntities without Code (geographical regions):")
print(df_co2_emissions[df_co2_emissions['Code'].isna()]['Entity'].unique())

# ============================================================================
# Step 2: Data Preparation
# ============================================================================
print("-"*80)
print("\n[Step 2] Data Preparation")

# Rename columns
df_co2_emissions_per_capita = df_co2_emissions_per_capita.rename(
    columns={'CO₂ emissions per capita': 'CO2 emissions per capita'}
)
df_co2_emissions = df_co2_emissions.rename(
    columns={'Annual CO₂ emissions': 'CO2 emissions'}
)

# Merge datasets
print("\nMerging datasets...")
df_co2_emissions_merged = pd.merge(
    df_co2_emissions_per_capita, 
    df_co2_emissions, 
    on=['Entity', 'Code', 'Year'], 
    how='outer'
)

# Filter years > 1800
df_co2_emissions_merged = df_co2_emissions_merged[df_co2_emissions_merged['Year'] > 1800]
print(f">>> Merged dataset shape: {df_co2_emissions_merged.shape}")

# Split into countries, continents, and income datasets
print("\nSplitting datasets...")

df_co2_emissions_countries = df_co2_emissions_merged[df_co2_emissions_merged['Code'].notna()]
print(f">>> Countries dataset: {df_co2_emissions_countries.shape}")
print(f"  Number of countries: {df_co2_emissions_countries['Entity'].nunique()}")

continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
df_co2_emissions_continents = df_co2_emissions_merged[df_co2_emissions_merged['Entity'].isin(continents)]
df_co2_emissions_continents.drop(columns=['Code'], inplace=True)
print(f">>> Continents dataset: {df_co2_emissions_continents.shape}")

income = ['High-income countries', 'Low-income countries',
          'Lower-middle-income countries', 'Upper-middle-income countries']
df_co2_emissions_income = df_co2_emissions_merged[df_co2_emissions_merged['Entity'].isin(income)]
df_co2_emissions_income.drop(columns=['Code'], inplace=True)
print(f">>> Income groups dataset: {df_co2_emissions_income.shape}")

# ============================================================================
# Step 3: Feature Understanding
# ============================================================================
print("-"*80)
print("\n[Step 3] Feature Understanding - Creating Visualizations")

# 3.1 CO2 emissions per Continent
print("\n3.1 Plotting CO2 emissions per Continent...")
fig, ax = plt.subplots(figsize=(14, 7))
entities = df_co2_emissions_continents['Entity'].unique()
colors = sns.color_palette("dark", len(entities))
for i, continent in enumerate(entities):
    data = df_co2_emissions_continents[df_co2_emissions_continents['Entity'] == continent]
    ax.plot(data['Year'], data['CO2 emissions'], marker='o', 
            label=continent, linewidth=2, markersize=4, color=colors[i])

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('CO2 emissions', fontsize=12, fontweight='bold')
ax.set_title('CO2 emissions Per Continent', fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/plots/co2_total_emissions_continents.png', dpi=300, bbox_inches='tight')
plt.close()
print(">>> Saved: co2_total_emissions_continents.png")
print("-"*80)

# 3.2 CO2 Emissions Per Capita Per Continent
print("\n3.2 Plotting CO2 Emissions Per Capita per Continent...")
fig, ax = plt.subplots(figsize=(14, 7))
for i, continent in enumerate(entities):
    data = df_co2_emissions_continents[df_co2_emissions_continents['Entity'] == continent]
    ax.plot(data['Year'], data['CO2 emissions per capita'], marker='o', 
            label=continent, linewidth=2, markersize=4, color=colors[i])

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('CO2 emissions per capita', fontsize=12, fontweight='bold')
ax.set_title('CO2 Emissions Per Capita Per Continent', fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/plots/co2_per_capita_continents.png', dpi=300, bbox_inches='tight')
plt.close()
print(">>> Saved: co2_per_capita_continents.png")
print("-"*80)

# 3.3 CO2 Emissions Per Capita Per Income Range
print("\n3.3 Plotting CO2 Emissions Per Capita per Income Range...")
fig, ax = plt.subplots(figsize=(14, 7))
income_entities = df_co2_emissions_income['Entity'].unique()
colors = sns.color_palette("inferno", len(income_entities))
for i, income_group in enumerate(income_entities):
    data = df_co2_emissions_income[df_co2_emissions_income['Entity'] == income_group]
    ax.plot(data['Year'], data['CO2 emissions per capita'], marker='s', 
            label=income_group, linewidth=2, markersize=5, color=colors[i])

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('CO2 emissions per capita', fontsize=12, fontweight='bold')
ax.set_title('CO2 Emissions Per Capita Per Income Range', fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/plots/co2_per_capita_income.png', dpi=300, bbox_inches='tight')
plt.close()
print(">>> Saved: co2_per_capita_income.png")

# ============================================================================
# Step 4: Data Saving
# ============================================================================
print("-"*80)
print("\n[Step 4] Saving Processed Data")

df_co2_emissions_countries.to_csv('../data/processed/co2_emissions_countries.csv', index=False)
print(">>> Saved: co2_emissions_countries.csv")

df_co2_emissions_continents.to_csv('../data/processed/co2_emissions_continents.csv', index=False)
print(">>> Saved: co2_emissions_continents.csv")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PROCESSING COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nDatasets created:")
print(f"  - Countries: {df_co2_emissions_countries.shape[0]} rows, {df_co2_emissions_countries.shape[1]} columns")
print(f"  - Continents: {df_co2_emissions_continents.shape[0]} rows, {df_co2_emissions_continents.shape[1]} columns")
print(f"  - Income groups: {df_co2_emissions_income.shape[0]} rows, {df_co2_emissions_income.shape[1]} columns")
print(f"\nPlots saved in: ../outputs/plots/")
print(f"Processed data saved in: ../data/processed/")
print("="*80)