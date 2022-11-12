import pandas as pd
import numpy as np
data_train = pd.read_csv('data/sdsh2022_sarafu_trainset.csv')
data_users = pd.read_csv('data/sdsh2022_sarafu_users.csv')
data_trxs = pd.read_csv('data/sdsh2022_sarafu_transactions.csv')
data_trxs['is_bonus'] = np.where(data_trxs['source'] == -1, 1, 0)
data_trxs['is_penalty'] = np.where(data_trxs['target'] == -1, 1, 0)
data_trxs['date'] = pd.to_datetime(data_trxs['time']).dt.date

stage_2_days = 60

data = data_train.merge(data_users, on='id', how='left')

data = data.merge(
    data_trxs.groupby(['source']).size().reset_index(name='n_transactions_sent'),
    how='left',
    left_on=['id'],
    right_on=['source']
).drop(columns=['source'])

data = data.merge(
    data_trxs.groupby(['target']).size().reset_index(name='n_transactions_received'),
    how='left',
    left_on=['id'],
    right_on=['target']
).drop(columns=['target'])

data = data.merge(
    data_trxs.groupby('source').sum().reset_index().loc[:, ['source', 'is_penalty']],
    how='left',
    left_on=['id'],
    right_on=['source']
).drop(columns=['source'])

data = data.merge(
    data_trxs.groupby('target').sum().reset_index().loc[:, ['target', 'is_bonus']],
    how='left',
    left_on=['id'],
    right_on=['target']
).drop(columns=['target'])

data = data.merge(
    data_trxs.groupby('target')['source'].nunique().reset_index(
    ).rename(columns={"target": "id", "source": "n_partners_received"}),
    how='left',
    on='id'
)

data = data.merge(
    data_trxs.groupby('source')['target'].nunique().reset_index(
    ).rename(columns={"source": "id", "target": "n_partners_sent"}),
    how='left',
    on='id'
)

data = data.merge(
    data_trxs.groupby('source')['date'].first().reset_index(
    ).rename(columns={"source": "id", "date": "first_trx_date"}),
    how='left',
    on='id'
)

data = data.merge(
    data_trxs.groupby('source')['date'].last().reset_index(
    ).rename(columns={"source": "id", "date": "last_trx_date"}),
    how='left',
    on='id'
)

data['n_transactions'] = data['n_transactions_received']+data['n_transactions_sent']
data['n_partners'] = data['n_partners_received']+data['n_partners_sent']
data['registration_date'] = pd.to_datetime(data['registration_time']).dt.date
data['days_before_first_trx'] = data['first_trx_date'] - data['registration_date']

data.rename(columns={"is_bonus": "n_bonuses", "is_penalty": "n_penalties"}, inplace=True)

data['r_time'] = pd.to_datetime(data['registration_time'])
data['days_from_registr'] = pd.to_datetime(data['r_time'].max()) - data['r_time']
data['days_from_registr'] = data['days_from_registr'] / np.timedelta64(1, 'D')
data.drop(columns=['r_time'], inplace=True)

data['count_intensity'] = data['n_transactions']/data['days_from_registr']
data['count_sent_intensity'] = data['n_transactions_sent']/data['days_from_registr']
data['count_received_intensity'] = data['n_transactions_received']/data['days_from_registr']

data['n_transactions_sent'] = data['n_transactions_sent'].fillna(0)
data['n_transactions_received'] = data['n_transactions_received'].fillna(0)
data['n_penalties'] = data['n_penalties'].fillna(0)
data['n_bonuses'] = data['n_bonuses'].fillna(0)
data['n_partners_received'] = data['n_partners_received'].fillna(0)
data['n_partners_sent'] = data['n_partners_sent'].fillna(0)
data['n_transactions'] = data['n_transactions'].fillna(0)
data['n_partners'] = data['n_partners'].fillna(0)
data['days_before_first_trx'] = data['days_before_first_trx'].astype('int64').fillna(0)
data['count_intensity'] = data['count_intensity'].fillna(0)
data['count_sent_intensity'] = data['count_sent_intensity'].fillna(0)
data['count_received_intensity'] = data['count_received_intensity'].fillna(0)

data['last_trx_date'] = data['last_trx_date'].fillna(
    pd.to_datetime(data['last_trx_date']).min()
)
data['first_trx_date'] = data['first_trx_date'].fillna(
    pd.to_datetime(data['first_trx_date']).max()
)

data['last_trx_date'] = pd.to_datetime(data['last_trx_date'])

data.drop(columns=['gender', 'area_type'], inplace=True)

data['days_past_last_trxs'] = data['last_trx_date'].max() - data['last_trx_date']
data['days_past_last_trxs'] = data['days_past_last_trxs'].dt.days

data['is_stage_2'] = np.where(data['days_past_last_trxs'] > stage_2_days, 1, 0)

data_new = data_trxs.merge(
    data.loc[:, ['id', 'is_stage_2']],
    how='left',
    left_on=['source'],
    right_on=['id']
)
data_new = data_new.loc[data_new['is_stage_2'] == 1, :].groupby('target')['source'].nunique(
).reset_index().rename(columns={"target": "id", "source": "n_partners_received_stage_2"})

data = data.merge(
    data_new,
    how='left',
    on='id'
)

data_new = data_trxs.merge(
    data.loc[:, ['id', 'is_stage_2']],
    how='left',
    left_on=['target'],
    right_on=['id']
)
data_new = data_new.loc[data_new['is_stage_2'] == 1, :].groupby('source')['target'].nunique(
).reset_index().rename(columns={"source": "id", "target": "n_partners_sent_stage_2"})

data = data.merge(
    data_new,
    how='left',
    on='id'
)

data['n_partners_received_stage_2'] = data['n_partners_received_stage_2'].fillna(0)
data['n_partners_sent_stage_2'] = data['n_partners_sent_stage_2'].fillna(0)
data['n_partners_stage_2'] = data['n_partners_sent_stage_2']+data['n_partners_received_stage_2']
data['stage_2_partners_share'] = data['n_partners_stage_2']/data['n_partners']

data['stage_2_partners_share'] = data['stage_2_partners_share'].fillna(0)
data['avg_time_between_trxs'] = data['days_from_registr']/data['n_transactions']

data = data.apply(lambda x: x.replace([np.inf, -np.inf], 0))
data = data.apply(lambda x: x.fillna(0))

data.to_csv('data/data_p2.csv', index=False)