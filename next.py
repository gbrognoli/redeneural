import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Dados de exemplo
textos = [
    "Aprendendo a programar em Python é muito",
    "Python é uma linguagem de programação",
    "Redes neurais artificiais são usadas para",
    "Programação em Python é",
    "O treinamento de uma rede neural requer"
]

# Preprocessamento dos dados
palavras = ' '.join(textos).split()
palavras_unicas = sorted(set(palavras))
indice_para_palavra = {indice: palavra for indice, palavra in enumerate(palavras_unicas)}
palavra_para_indice = {palavra: indice for indice, palavra in enumerate(palavras_unicas)}
tamanho_vocabulario = len(palavras_unicas)
sequencias = []
proxima_palavra = []
tamanho_sequencia = 3

for i in range(len(palavras) - tamanho_sequencia):
    sequencia = palavras[i:i + tamanho_sequencia]
    sequencias.append([palavra_para_indice[palavra] for palavra in sequencia])
    proxima_palavra.append(palavra_para_indice[palavras[i + tamanho_sequencia]])

# Convertendo para arrays numpy
X = np.array(sequencias)
y = np.array(proxima_palavra)

# Criando o modelo RNN mais capaz
modelo = Sequential()
modelo.add(Embedding(tamanho_vocabulario, 100, input_length=tamanho_sequencia))
modelo.add(LSTM(256, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(128))
modelo.add(Dropout(0.2))
modelo.add(Dense(tamanho_vocabulario, activation='softmax'))

# Compilando o modelo
modelo.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
modelo.fit(X, y, epochs=200, batch_size=64)

# Função para prever a próxima palavra
def prever_proxima_palavra(frase):
    palavras_frase = frase.split()[-tamanho_sequencia:]
    sequencia = [palavra_para_indice[palavra] for palavra in palavras_frase]
    sequencia = np.array([sequencia])
    indice_palavra_prevista = modelo.predict_classes(sequencia, verbose=0)[0]
    palavra_prevista = indice_para_palavra[indice_palavra_prevista]
    return palavra_prevista

# Exemplo de uso
frase_exemplo = "Python é uma linguagem de"
proxima_palavra_prevista = prever_proxima_palavra(frase_exemplo)
print(f"Próxima palavra prevista: {proxima_palavra_prevista}")
