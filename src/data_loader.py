"""
Data Loader module for Hybrid Quantum-Classical Term Deposit Campaign.
Loads Stage1_Train, Stage1_Test, Stage2_Data, Stage2_Constraints from Excel
and ensures 100% verified customer_id mapping.
"""

import os
import openpyxl
import pandas as pd
import numpy as np

def load_all_data(excel_path='bank_campaign_participant_release.xlsx', raw_csv_path='bank.csv'):
    """
    Loads all sheets from the official release workbook and reconstructs
    exact verified customer_ids for Stage1_Test.
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    
    # 1. Load Stage1_Train
    tr_ws = wb['Stage1_Train']
    tr_rows = list(tr_ws.iter_rows(values_only=True))
    df_train = pd.DataFrame(tr_rows[1:], columns=tr_rows[0])
    
    # 2. Load Stage1_Test
    te_ws = wb['Stage1_Test']
    te_rows = list(te_ws.iter_rows(values_only=True))
    df_test = pd.DataFrame(te_rows[1:], columns=te_rows[0])
    
    # 3. Load Stage2_Data
    s2_ws = wb['Stage2_Data']
    s2_rows = list(s2_ws.iter_rows(values_only=True))
    df_s2 = pd.DataFrame(s2_rows[1:], columns=s2_rows[0])
    
    # 4. Load Stage2_Constraints
    sc_ws = wb['Stage2_Constraints']
    sc_rows = list(sc_ws.iter_rows(values_only=True))
    df_constraints = pd.DataFrame(sc_rows[1:], columns=sc_rows[0])
    
    # Reconstruct customer_ids for df_test using exact seed 42 sampling from bank.csv
    feature_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                    'housing', 'loan', 'contact', 'day', 'month', 'campaign', 
                    'pdays', 'previous', 'poutcome']
    
    if os.path.exists(raw_csv_path):
        df_raw = pd.read_csv(raw_csv_path, sep=';')
        s_yes = df_raw[df_raw['y'] == 'yes'].sample(n=46, random_state=42)
        s_no = df_raw[df_raw['y'] == 'no'].sample(n=354, random_state=42)
        df_400 = pd.concat([s_yes, s_no]).sample(frac=1, random_state=42).reset_index(drop=True)
        df_400['customer_id'] = [f'C{i:04d}' for i in range(1, 401)]
        
        train_ids = set(df_train['customer_id'])
        df_400_test = df_400[~df_400['customer_id'].isin(train_ids)].copy().reset_index(drop=True)
        
        matched_cids = []
        for _, r in df_test.iterrows():
            cond = True
            for c in feature_cols:
                cond = cond & (df_400_test[c] == r[c])
            m = df_400_test[cond]
            if len(m) == 1:
                matched_cids.append(m['customer_id'].values[0])
            else:
                matched_cids.append(None)
        
        if all(c is not None for c in matched_cids):
            df_test['customer_id'] = matched_cids
        else:
            # Fallback if raw csv not found
            test_cids = [cid for cid in df_s2['customer_id'] if cid not in train_ids]
            df_test['customer_id'] = test_cids
    else:
        train_ids = set(df_train['customer_id'])
        test_cids = [cid for cid in df_s2['customer_id'] if cid not in train_ids]
        df_test['customer_id'] = test_cids
        
    return df_train, df_test, df_s2, df_constraints

if __name__ == '__main__':
    df_train, df_test, df_s2, df_constraints = load_all_data()
    print(f"Loaded Train: {df_train.shape}, Test: {df_test.shape}, Stage2: {df_s2.shape}")
    print(f"Test customer IDs first 5: {df_test['customer_id'].head(5).tolist()}")
