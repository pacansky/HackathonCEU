# PRESETS ##############################################################################################
import pandas as pd
import numpy as np


# BEFORE EDA ###########################################################################################
data_train = pd.read_csv('data/sdsh2022_sarafu_trainset.csv')
data_users = pd.read_csv('data/sdsh2022_sarafu_users.csv')
data_trxs = pd.read_csv('data/sdsh2022_sarafu_transactions.csv')
data_trxs['is_bonus'] = np.where(data_trxs['source'] == -1, 1, 0)
data_trxs['is_penalty'] = np.where(data_trxs['target'] == -1, 1, 0)
data_trxs['date'] = pd.to_datetime(data_trxs['time']).dt.date

stage_2_days = 30

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
data['days_before_first_trx'] = data['days_before_first_trx'].dt.days.fillna(0)

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


#BEFORE EDA SECOND ########################################################################################
data.to_csv('data/data_p2.csv', index=False)


# AFTER EDA ###############################################################################################
data = pd.read_csv('data/data_p2.csv')

data['start_balance_log'] = np.log(data['start_balance']).replace([np.inf, -np.inf], 0)
data['n_transactions_log'] = np.log(data['n_transactions']).replace([np.inf, -np.inf], 0)
data['n_transactions_received_log'] = np.log(data['n_transactions_received']
                                            ).replace([np.inf, -np.inf], 0)
data['n_transactions_sent_log'] = np.log(data['n_transactions_sent']).replace([np.inf, -np.inf], 0)
data['n_bonuses_log'] = np.log(data['n_bonuses']).replace([np.inf, -np.inf], 0)
#data['n_penalties'] = data['start_balance']
data['n_partners_log'] = np.log(data['n_partners']).replace([np.inf, -np.inf], 0)
data['n_partners_stage_2_log'] = np.log(data['n_partners_stage_2']).replace([np.inf, -np.inf], 0)
data['n_partners_received_log'] = np.log(data['n_partners_received']).replace([np.inf, -np.inf], 0)
data['n_partners_received_stage_2_log'] = np.log(data['n_partners_received_stage_2']
                                                ).replace([np.inf, -np.inf], 0)
data['n_partners_sent_log'] = np.log(data['n_partners_sent']).replace([np.inf, -np.inf], 0)
data['n_partners_sent_stage_2_log'] = np.log(data['n_partners_sent_stage_2']
                                            ).replace([np.inf, -np.inf], 0)
data['days_before_first_trx_log'] = np.log(data['days_before_first_trx']
                                          ).replace([np.inf, -np.inf], 0)
data['count_received_intensity_log'] = np.log(data['count_received_intensity']
                                          ).replace([np.inf, -np.inf], 0)
data['count_sent_intensity_log'] = np.log(data['count_sent_intensity']
                                          ).replace([np.inf, -np.inf], 0)
data['count_intensity_log'] = np.log(data['count_intensity']).replace([np.inf, -np.inf], 0)

data['n_partners_ratio'] = data['n_partners_received']/data['n_partners_sent']
data['n_partners_ratio_log'] = data['n_partners_received_log']/data['n_partners_sent_log']

data['n_transactions_ratio'] = data['n_transactions_received']/data['n_transactions_sent']
data['n_transactions_ratio_log'] = (
    data['n_transactions_received_log']/data['n_transactions_sent_log']
)

data['count_intensity_ratio'] = data['count_received_intensity']/data['count_sent_intensity']
data['count_intensity_ratio_log'] = (
    data['count_received_intensity_log']/data['count_sent_intensity_log']
)

data['n_partners_stage_2_ratio'] = (data['n_partners_received_stage_2']/
                                    data['n_partners_sent_stage_2'])
data['n_partners_stage_2_ratio_log'] = (data['n_partners_received_stage_2_log']/
                                        data['n_partners_sent_stage_2_log'])
#data_penalties = np.
####
# amount_received
transactions_received = data_trxs.groupby('target')['amount'].sum().reset_index()
transactions_received = transactions_received.rename(columns={'amount': 'amount_received',
                                                              'target': 'id'})
data = data.merge(transactions_received,
                  on='id', how='left')

# amount_sent
transactions_sent = data_trxs.groupby('source')['amount'].sum().reset_index()
transactions_sent = transactions_sent.rename(columns={'amount': 'amount_sent',
                                                      'source': 'id'})
data = data.merge(transactions_sent,
                  on='id', how='left')
# amount_total
data['amount_total'] = data['amount_sent'] + data['amount_received']

# amount_bonuses
transactions_bonuses = data_trxs[data_trxs['source'] == -1
                                ].groupby('target')['amount'].sum().reset_index()
transactions_bonuses = transactions_bonuses.rename(columns={'amount': 'amount_bonuses',
                                                            'target': 'id'})
data = data.merge(transactions_bonuses,
                  on='id', how='left')

# amount_penalties
transactions_penalties = data_trxs[data_trxs['target'] == -1
                                  ].groupby('source')['amount'].sum().reset_index()
transactions_penalties = transactions_penalties.rename(columns={'amount': 'amount_penalties',
                                                                'source': 'id'})
data = data.merge(transactions_penalties,
                  on='id', how='left')

# balance_growth
data['balance_growth'] = data['start_balance'] + data['amount_received'] - data['amount_sent']

# amount_intensity
data['amount_intensity'] = data['amount_total'] / data['days_from_registr']

# amount_sent_intensity
data['amount_sent_intensity'] = data['amount_sent'] / data['days_from_registr']

# amount_received_intensity
data['amount_received_intensity'] = data['amount_received'] / data['days_from_registr']

col_to_transform = ['amount_received_intensity', 'amount_sent_intensity',
    'amount_intensity', 'balance_growth', 'amount_penalties', 'amount_bonuses',
    'amount_total', 'amount_sent', 'amount_received']
####

for t, c in zip(data[col_to_transform].dtypes, data[col_to_transform].columns):
    if t in ('int64', 'float64'):
        data[c] = (data[c] - min(data[c]))/(max(data[c]) - min(data[c])) + 1e-6
        data[c+'_log'] = np.log(data[c]).replace([np.inf, -np.inf], 0)
####
# 'business_type', 'area_name'
to_onehot = ['business_type',
             'area_name', 'area_type', 'gender']

for c in to_onehot:
    for i, k in data[c].value_counts(normalize=True).items():
        if k > 0.1:
            data[i] = np.where(data[c] == i, 1, 0)
    if all(data[c].value_counts(normalize=True) > 0.1):
        col_to_drop = data[c].value_counts().sort_values(ascending=True).index[0]
        data.drop(columns=[col_to_drop], inplace=True)
####

cols_to_drop = [
    'id', 'start_balance',
    'n_bonuses', 'n_penalties',
    'n_transactions',
    'n_transactions_received', 'n_transactions_sent',
    'n_transactions_received_log', 'n_transactions_sent_log',
    'count_received_intensity', 'count_sent_intensity',
    'n_partners', 'n_partners_stage_2',
    'days_before_first_trx',

    'n_partners_received',
    'n_partners_received', 'n_partners_sent',
    'n_partners_received_log', 'n_partners_sent_log',
    'n_partners_received_stage_2', 'n_partners_sent_stage_2',
    'n_partners_received_stage_2_log', 'n_partners_sent_stage_2_log',

    'count_received_intensity_log', 'count_sent_intensity_log',
    'n_partners_ratio', 'n_transactions_ratio', 'count_intensity_ratio',
    'n_partners_stage_2_ratio', 'count_intensity',

    'registration_time', 'first_trx_date', 'last_trx_date',
    'registration_date', 'days_from_registr',
    'account_type', 'business_type', 'area_name', 'area_type', 'gender'
]
data.drop(columns=cols_to_drop, inplace=True)
data.drop(columns=col_to_transform, inplace=True)
data = data.apply(lambda x: x.fillna(0))
data = data.apply(lambda x: x.replace([np.inf, -np.inf], 0))

# AFTER EDA SECOND #########################################################################################
data.to_csv('data/data_p3.csv', index=False)
