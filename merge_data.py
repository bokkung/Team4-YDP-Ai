import pandas as pd
from pathlib import Path

# 1. Define File Paths (ใช้เครื่องหมาย = และ quotes r'...' ให้ถูกต้อง)
ASSET_CSV = Path(r'C:\Users\bokthaiban\Desktop\mercilnew\data\assets_rows_with_fixed_types.csv')
POI_CSV = Path(r'C:\Users\bokthaiban\Desktop\mercilnew\poi_results_enhanced.csv')
OUTPUT_CSV = Path(r'C:\Users\bokthaiban\Desktop\mercilnew\data\assets_rows_merged_with_poi.csv')

# 2. Read DataFrames (ใช้เครื่องหมาย = และระบุค่าใน fillna())
df_assets = pd.read_csv(ASSET_CSV).fillna('')
df_poi = pd.read_csv(POI_CSV).fillna('')

# 3. Print Status (ใช้ f'' quotes ให้ถูกต้อง)
print(f'Assets: {len(df_assets)} rows')
print(f'POI: {len(df_poi)} rows')

# 4. Merge and Save (ใช้ quotes สำหรับ argument ที่เป็น string: on='id', how='left')
df_merged = df_assets.merge(df_poi, on='id', how='left')

df_merged.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f'\nSuccessfully merged {len(df_merged)} rows to: {OUTPUT_CSV}')