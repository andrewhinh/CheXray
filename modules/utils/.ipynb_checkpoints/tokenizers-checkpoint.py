#import multiprocessing
import re
from collections import Counter
"""
Functions for preprocessing text data and adding tokenized/numericalized columns to original csv
"""

#Preprocessing
def clean_report(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

#Rep Gen
def add_pad(df): df['tok_reports'] = 'xxpad ' + df['reports'] + ' xxpad'
def add_tok_col(df): df['tok_reports'] = df['tok_reports'].str.split()
def add_len_col(df): df['tok_reports_length'] = df['tok_reports'].str.len()
#def add_tok_col(df): return tokenize_df(df, 'reports', n_workers=multiprocessing.cpu_count(), tok_text_col='tok_reports')
#def add_end(df): df[0]['tok_reports'] += ['xxend']
#def add_one(df): df[0]['tok_reports_length'] += 1

def to_max(df, max_seq_len): df.loc[df['tok_reports_length']>max_seq_len, 'tok_reports_length']=max_seq_len 
#def to_max(df, max_seq_len): df[0].loc[df[0]['tok_reports_length']>max_seq_len, 'tok_reports_length']=max_seq_len
    
def make_vocab_counter(df):
    total_tokens = []
    for idx, row in df.iterrows():
        for token in row.loc['tok_reports']: total_tokens.append(token)
    return Counter(total_tokens)    
  
def temp_add(row, max_seq_len, vocab):
    temp = []
    report = row.loc['tok_reports'][:max_seq_len]
    for word in report:
        try: temp.append(vocab.index(word))
        except: temp.append(vocab.index('xxunk'))
    return temp
"""
def temp_add(row, max_seq_len, vocab):
    temp = []
    report = row.loc['tok_reports'][:max_seq_len]
    for word in report:
        try: temp.append(vocab.index(word))
        except: temp.append(vocab.index('xxunk'))
    return temp
"""
def add_text_to_num(df, max_seq_len, vocab): df['idx_reports'] = df.apply(temp_add, args=(max_seq_len, vocab,), axis=1)

def add_ones(row): return [1]*row['tok_reports_length']
def add_mask(df):  df['mask_reports'] = df.apply(add_ones, axis=1)  
#def add_mask(df): df[0].insert(len(df[0].columns), 'mask_reports', np.nan, True)   
#def add_ones(row): return [1]*row['tok_reports_length']
#def apply_funcs(df): df[0]['mask_reports'] = df[0].apply(add_ones, axis=1)