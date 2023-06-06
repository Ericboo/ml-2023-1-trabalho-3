import pandas as pd
import numpy as np
import rede_neural as rn
import seaborn as sns
from normalizar import Pre_processamento
from matplotlib import pyplot as plt

"""
def plot_compare(x, y1, y2, color1: str, color2: str, label1: str, label2: str, 
                 title: str, xlabel: str, ylabel: str):
    x = pd.DataFrame(x)
    y1 = pd.DataFrame(y1)
    y2 = pd.DataFrame(y2)

    # Redefinir os índices para evitar erros de alinhamento
    x = x.reset_index(drop=True)
    y1 = y1.reset_index(drop=True)
    y2 = y2.reset_index(drop=True)

    # Ordenar os valores de salário e ajustar as correspondentes observações em x
    sorted_indices = np.argsort(y1.values.flatten())
    sorted_x = x.values[sorted_indices]
    sorted_y1 = y1.values[sorted_indices]
    sorted_y2 = y2.values[sorted_indices]

    plt.figure()
    plt.plot(sorted_x, sorted_y1, color=color1, label=label1)
    plt.plot(sorted_x, sorted_y2, color=color2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""

def plot_compare(y_test, y_pred):
    df = pd.DataFrame({'Dado Real': y_test.flatten(), 'Dado Predito': y_pred.flatten()})

    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df)
    plt.title('Resultado da Predição comparado com Dado Real')
    plt.xlabel('Observação')
    plt.ylabel('Salário')
    plt.tight_layout()
    plt.show()


df = pd.read_csv('ds_salaries.csv')
pre_processador = Pre_processamento(df)
df = pre_processador.substitute_str(df)
df = pre_processador.drop_too_little_data(df, col='job_title')

# Extraia a coluna 'job_title'
job_titles = df['job_title']
unique_job_titles = job_titles.unique()
num_rows_per_title = 10
selected_data = []

# Para cada tipo de job title, selecione aleatoriamente o número desejado de linhas
for title in unique_job_titles:
    selected_rows = df[df['job_title'] == title].sample(num_rows_per_title)
    selected_data.append(selected_rows)

x_train = pd.concat(selected_data)
x_train = x_train.sample(frac=1).reset_index(drop=True)
y_train = x_train['salary_in_usd']
x_train = x_train.drop('salary_in_usd', axis=1)
x_train = x_train.drop('salary', axis=1)
 
x_test = df.sample(frac=1).reset_index(drop=True)
y_test = x_test['salary_in_usd']
x_test = x_test.drop('salary_in_usd', axis=1)
x_test = x_test.drop('salary', axis=1)

model = rn.rede_neural()
history = model.fit(x_train, 
                    y_train,
                    batch_size=12,
                    epochs=800)

y_pred = model.predict(x_test)

df['index'] = range(len(df))


# Plotagem do gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valor Real')
plt.scatter(range(len(y_pred)), y_pred, color='red', marker='x', label='Valor Previsto')
plt.xlabel('ID')
plt.ylabel('Salário')
plt.legend()
plt.title('Gráfico de Dispersão - Valores Reais vs. Valores Previstos')
plt.show()