import pandas as pd
import numpy as np
import rede_neural as rn
from normalizar import Pre_processamento
from matplotlib import pyplot as plt

df = pd.read_csv('ds_salaries.csv')
pre_processador = Pre_processamento(df)
df = pre_processador.substitute_str(df)
df = pre_processador.drop_too_little_data(df, col='job_title', min_val=20)

# Extraia a coluna 'job_title'
job_titles = df['job_title']
unique_job_titles = job_titles.unique()
num_rows_per_title = 22
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
                    batch_size=24,
                    epochs=800)

y_pred = model.predict(x_test)

df['index'] = range(len(df))
selecionados = np.random.choice(df['index'], size=500, replace=False)
y_test_selecionados = []
y_pred_selecionados = []
for selecionado in selecionados:
    y_test_selecionados.append(y_test[selecionado])
    y_pred_selecionados.append(y_pred[selecionado])


# Plotagem do gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test_selecionados)), y_test_selecionados, color='blue', label='Valor Real')
plt.scatter(range(len(y_pred_selecionados)), y_pred_selecionados, color='red', marker='x', label='Valor Previsto')
plt.xlabel('ID')
plt.ylabel('Salário')
plt.legend()
plt.title('Gráfico de Dispersão - Valores Reais vs. Valores Previstos')
plt.show()