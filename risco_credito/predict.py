# ========= bibliotecas =============
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import r2_score # para medir a acuracia do modelo
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler # para padronizar os dados
from sklearn.impute import SimpleImputer

import time
import joblib # para salvar o modelo preditivo
clf = joblib.load('./modelo/modelo_new_clf_treinado.pk')

sns.set()
#%matplotlib inline
warnings.filterwarnings('ignore')

# carrega o arquivo
arquivo = './datasets/credit_risk_dataset.csv'



# tratamento e limpeza de arquivo
def load_df(arquivo):
    df = pd.read_csv(arquivo)
    print('Arquivos recebidos...\nTratando os arquivos...')

    # converte colunas para numerica
    df.person_emp_length = pd.to_numeric(df.person_emp_length, errors='coerce')
    df.loan_int_rate = pd.to_numeric(df.loan_int_rate, errors='coerce')

    # preenche nulos com a media e mediana
    df.person_emp_length = df.person_emp_length.fillna(df.person_emp_length.mean())
    df.loan_int_rate = df.loan_int_rate.fillna(df.loan_int_rate.median())

    # extrai amostra considerando menor idade de 94 e tempo de emprego menor que 120
    df = df[(df['person_age'] <= 94) & (df['person_emp_length'] < 120)]

    # # Definir os limites para cada faixa etária
    age_bins = [18, 30, 40, 50, 60, 100]
    age_labels = ['18-30', '31-40', '41-50', '51-60', '61+']

    # adicionando nova coluna ao dataframe
    df['person_age_group'] = pd.cut(df['person_age'], bins=age_bins, labels=age_labels, right=False)

    print('Arquivos carregados e tratados com sucesso...')

    return df

# pre-processamento e separação dos dados
def createXy(df, input_columns, output_column):

    print('Preparando os arquivos para o processamento...')
    minmax = MinMaxScaler()
    onehot = OneHotEncoder(sparse=False, drop='first')
    sm = SMOTE(random_state=42)

    df.dropna(axis=0, inplace=True)
    X = df[input_columns].copy()
    X_num = X.select_dtypes(exclude=['object', 'category'])
    X_num = minmax.fit_transform(X_num)
    X_num = pd.DataFrame(data=X_num, columns=X.select_dtypes(exclude=['object', 'category']).columns)

    X_cat = X.select_dtypes(include=['object', 'category'])
    X_cat = onehot.fit_transform(X_cat)
    columns = []
    for c, values in zip(X.select_dtypes(include=['object', 'category']).columns, onehot.categories_):
        for value in values[1:]:
            columns.append(f'{c}_{value}')
    X_cat = pd.DataFrame(data=X_cat, columns=columns)
    X = X_num.join(X_cat)
    y = df[output_column].copy()
    print('Arquivos processados com sucesso...')

    return X, y

# previsao e adiciona colunas ao novo dataset
def df_proba(X, df):
    previsoes = clf.predict(X)
    probabilidades = clf.predict_proba(X)
    df['predict'] = previsoes
    df['probability'] = probabilidades[:, 1]
    print('Novo DataFrame gerado com sucesso...\n')
    return df

def create_df():
    # carrega e transforma o arquivo
    df = load_df(arquivo)
    # processa os dados
    X, y = createXy(df, ['person_age', 'person_income', 'person_home_ownership',
        'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file', 'cb_person_cred_hist_length',
        'person_age_group'], 'loan_status')
    df_new_pred = df_proba(X, df)
    df_new_pred.to_csv('./datasets/df_new_pred.csv', index=False)
    print('Arquivo .csv gerado com sucesso')

    return None


def main():
    create_df()

if __name__ == '__main__':
    main() 