import pandas as pd
import numpy as np

PATH = '../data/Limonene_data_for_ART.csv'
RESPONSE_VARS = ['Limonene']
INPUT_VARS = ['ATOB_ECOLI','ERG8_YEAST','IDI_ECOLI',
                   'KIME_YEAST','MVD1_YEAST','Q40322_MENSP',
                   'Q8LKJ3_ABIGR','Q9FD86_STAAU','Q9FD87_STAAU']
DBTL_A = ['2X-Mh', 'B-Lm', '2X-Ll', 'A-Mm', 'B-Ll', 'A-Mh', '2X-Lm',
       'A-Hl', '2X-Hh', 'B-Ml', 'B-Mm', '2X-Lh', 'B-Mh', '2X-Hl', 'B-Hl',
       '2X-Ml', 'B-Hm', 'B-Lh', 'B-Hh', 'A-Ll', 'A-Hm', '2X-Mm', 'A-Hh',
       'A-Ml', 'A-Lm',  'A-Lh', '2X-Hm']
DBTL_B = ['BL-Mm', 'BL-Mh', 'BL-Ml']

def read_data(path: str):
    ''' Read in data from csv file '''
    df = pd.read_csv(path, index_col=0)
    df = df[['Line Name', 'Type', '24']] # Keep only columns EDD style
    df = df.rename(columns={'24': 'value'}) # Rename 24.0 to value
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ''' Get data to the right format for analysis '''
    new_columns = df['Type'].unique()
    data = pd.DataFrame()
    data.index = df['Line Name'].unique()
    # add new columns to data
    for col in new_columns:
        data[col] = 0
    # fill in data
    for l in data.index:
        for c in new_columns:
            value = df[(df['Line Name'] == l) & (df['Type'] == c)]['value'].values
            data.loc[l, c] = value
    # drop OD column
    data.drop('Optical Density', axis=1, inplace=True)
    return data

def main():
    ''' Read original data and transform it for analysis'''
    df = read_data(PATH)
    print(f'Original data shape: {df.shape}')
    data = transform_data(df)
    print(f'Processed data shape: {data.shape}')
    data.rename(columns={'4-isopropenyl-1-methyl-cyclohexene':'Limonene'}, inplace=True)
    data.to_csv('../data/preprocessed_Limonene_data.csv')
    
if __name__ == '__main__':
    main()