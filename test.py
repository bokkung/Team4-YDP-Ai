import pandas as pd 

df = pd.read_csv(r"C:\Users\bokthaiban\Desktop\mercilnew\assets_rows_merged_with_poi.csv")
d = pd.read_csv(r"C:\Users\bokthaiban\Desktop\mercilnew\poi_results_enhanced.csv")
# print(df.columns.tolist())
print(d['university_name'])