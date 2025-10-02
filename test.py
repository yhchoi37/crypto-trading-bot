import pandas as pd

all_data = []
cache_file = '/config/workspace/data_cache/ETH_minute60_2022-01-01_to_2025-06-30.parquet'
df1 = pd.read_parquet(cache_file)
all_data.append(df1)
cache_file = '/config/workspace/data_cache/BTC_minute60_2022-01-01_to_2025-06-30.parquet'
df2 = pd.read_parquet(cache_file)
all_data.append(df2)

final_df = pd.concat(all_data, ignore_index=True)
if 'timestamp' in final_df.columns:
        final_df.rename(columns={'timestamp': 'index'}, inplace=True)
print(final_df.head())
print(final_df.tail())