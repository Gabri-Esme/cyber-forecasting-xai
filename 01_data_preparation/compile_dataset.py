import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer
from functools import reduce

load_dotenv("01_data_preparation/secrets.env")
DB_NAME = os.getenv("DB_PATH")
conn = sqlite3.connect(DB_NAME)

table_names = ['youtube_data', 'holidays_counter', 'monthly_attacks', 'monthly_pats']
dfs = {}

for table in table_names:
    dfs[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)

conn.close()

youtube_df = dfs['youtube_data']
holidays_df = dfs['holidays_counter']
attacks_df = dfs['monthly_attacks']
pats_df = dfs['monthly_pats']

# Youtube Dataset
youtube_df = youtube_df.groupby(['year', 'month']).agg(
    youtube_views=('views', 'sum'),
    avg_youtube_views=('views', 'mean')
).reset_index()


# Holiday Data
holidays_df = holidays_df.groupby(['year', 'month']).agg(
    Holidays=('holiday_count', 'sum')).reset_index()

# Monthly PATs
pats_df = pats_df.pivot(index=['year', 'month'], columns="pat_type", values='count').reset_index()
col_mapping = {
    'Application Whitelisting': 'Solution_APPLICATION WHITELISTING_Mentions',
    'Attack Tree': 'Solution_ATTACK TREE_Mentions',     
    'Behaviour-based Detection': 'Solution_BEHAVIOR BASED DETECTION_Mentions',
    'Blackholing': 'Solution_BLACKHOLING_Mentions', 
    'Blacklisting': 'Solution_BLACKLISTING_Mentions',
    'Code Signing': 'Solution_CODE SIGNING_Mentions', 
    'Continuous Authentication': 'Solution_CONTINUOUS AUTHENTICATION_Mentions', 
    'Control Flow Integrity': 'Solution_Control Flow Integrity_Mentions',
    'Darknet Monitoring': 'Solution_DARKNET MONITORING_Mentions', 
    'Data Leakage Detection/Prevention': 'Solution_DATA LEAKAGE DETECTION/PREVENTION_Mentions',
    'Data Loss Prevention': 'Solution_DATA LOSS PREVENTION_Mentions', 
    'Data Provenance': 'Solution_DATA PROVENANCE_Mentions', 
    'Deception Technology': 'Solution_DECEPTION TECHNOLOGY_Mentions',
    'Dynamic Analysis': 'Solution_DYNAMIC ANALYSIS_Mentions', 
    'Dynamic Resource Management': 'Solution_DYNAMIC RESOURCE MANAGEMENT_Mentions',
    'File Integrity Monitoring': 'Solution_FILE INTEGRITY MONITORING_Mentions', 
    'Formal Verification': 'Solution_FORMAL VERIFICATION_Mentions',
    'Graphical Authentication': 'Solution_GRAPHICAL AUTHENTICATION_Mentions', 
    'Honeypot': 'Solution_HONEYPOT_Mentions', 
    'Hypergame': 'Solution_HYPERGAME_Mentions',
    'Identity Management': 'Solution_IDENTITY MANAGEMENT_Mentions', 
    'Identity-based Encryption': 'Solution_Identity-Based Encryption (IBE)_Mentions',
    'Intrusion Detection/Prevention System': 'Solution_INTRUSION DETECTION/PREVENTION SYSTEM_Mentions', 
    'Keystroke Dynamics': 'Solution_KEYSTROKE DYNAMICS_Mentions',
    'Least Privilege': 'Solution_LEAST PRIVILEGE_Mentions', 
    'Moving Target Defence': 'Solution_MOVING TARGET DEFENSE_Mentions',
    'Multi-factor Authentication': 'Solution_MULTI FACTOR AUTHENTICATION_Mentions', 
    'Mutual Authentication': 'Solution_MUTUAL AUTHENTICATION_Mentions',
    'Network Segmentation': 'Solution_NETWORK SEGMENTATION_Mentions', 
    'One Time Password': 'Solution_ONE TIME PASSWORD_Mentions', 
    'Packet Filtering': 'Solution_PACKET FILTERING_Mentions',
    'Password Hashing': 'Solution_PASSWORD HASHING_Mentions', 
    'Password Management': 'Solution_PASSWORD MANAGEMENT_Mentions', 
    'Patch Management': 'Solution_PATCH MANAGEMENT_Mentions',
    'Public Key Infrastructure': 'Solution_PUBLIC KEY INFRASTRUCTURE_Mentions', 
    'Statistical Hidden Markov Model': 'Solution_HIDDEN MARKOV MODEL_Mentions'
    }
pats_df = pats_df.rename(columns=col_mapping)

# Monthly Attacks
attacks_df = attacks_df.pivot(index=['year', 'month'], columns="attack_type", values='count').reset_index()
col_mapping = {
    'APT': 'Mentions-Advanced persistent threat',
    'Account Hijacking': 'Mentions-Account Hijacking', 
    'Backdoor': 'Mentions-Backdoor', 
    'Botnet': 'Mentions-Botnet', 
    'DDoS': 'Mentions-DDoS',
    'Disinformation': 'Mentions-Disinformation/Misinformation', 
    'Dropper': 'Mentions-Dropper', 
    'Insider Threat': 'Mentions-Insider Threat', 
    'Malware': 'Mentions-Malware',
    'Password Attack': 'Mentions-Password Attack',
    'Phishing': 'Mentions-Phishing',
    'Ransomware': 'Mentions-Ransomware',
    'Session Hijacking': 'Mentions-Session Hijacking',
    'Targeted Attack': 'Mentions-Targeted Attack', 
    'Trojan': 'Mentions-Trojan',
    'Vulnerability': 'Mentions-Vulnerability',
    'Zero-day': 'Mentions-Zero-day'
    }
attacks_df = attacks_df.rename(columns=col_mapping)

# Merge DFs
merge_key = ["year", 'month']
merged1_df = reduce(
    lambda left, right: pd.merge(left, right, on=merge_key, how="outer"),
    [youtube_df, holidays_df, pats_df, attacks_df]
)

merged1_df['time'] = pd.to_datetime(merged1_df[['year', 'month']].assign(day=1)).dt.to_period('M')
merged1_df = merged1_df.drop(columns=['year', 'month'])

# Add Incident Data
incident_df = pd.read_csv('01_data_preparation/incident_data/output_data/incidents_clustered.csv')
cluster_cols = [col for col in incident_df.columns if col.startswith('cluster_')]
incident_df = incident_df[['start_date'] + cluster_cols]
incident_df['time'] = pd.to_datetime(incident_df['start_date'], errors='coerce').dt.to_period('M')
incident_df = incident_df.drop(columns=['start_date'])
incident_df = incident_df.groupby('time', as_index=False).sum()

merged2_df = reduce(
    lambda left, right: pd.merge(left, right, on='time', how="outer"),
    [merged1_df, incident_df]
)


# Merge old dataset with new
old_df = pd.read_csv('01_data_preparation/data/old_data.csv')
old_df['time'] = pd.to_datetime(old_df['Date'], format='%b-%y', errors='coerce').dt.to_period('M')
old_df = old_df.drop(columns=['Date']).set_index('time')

merged2_df = merged2_df.set_index('time')

merged3_df = merged2_df.combine_first(old_df)

merged3_df = merged3_df[merged3_df.index >= pd.Period('2012-01', 'M')]
merged3_df = merged3_df[merged3_df.index <= pd.Period('2026-08', 'M')]
merged3_df = merged3_df.drop(columns=['HTTPS', 'Password Policy', 'Password Strength Meters', 'Split Manufacturing'])
merged3_df = merged3_df.sort_index(ascending=True)


# Impute Data to fill missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(merged3_df)

imputed_df = pd.DataFrame(imputed_data, columns=merged3_df.columns, index=merged3_df.index)

imputed_df.to_csv('02_model/data/data.csv', index=True)
