# For new data_path for full data set, combining the trainset(proto_train) and validation set(proto_val) into one csv
import pandas as pd

train_df = pd.read_csv('data/proto_train_final.csv')
val_df = pd.read_csv('data/proto_val_final.csv')

# Combine them together
combined_df = pd.concat([train_df, val_df], ignore_index=True)
combined_df.to_csv('data/proto_train_val_final.csv', index=False)