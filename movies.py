import pandas as pd
import numpy as np

# -------------------- Funções de Manipulação & Preparação dos Dados --------------------

def carregar_dados(ratings_path='ratings.csv', movies_path='movies.csv'):
    """Carrega e retorna os datasets de ratings e movies."""
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

def tratar_valores_nulos(df):
    """Remove valores nulos de um DataFrame."""
    df.dropna(inplace=True)
    return df

def converter_timestamp(ratings):
    """Converte o campo timestamp para uma data legível."""
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    return ratings

def criar_matriz_usuario_filme(ratings):
    """Cria e retorna uma matriz de usuários-filmes preenchendo valores nulos com 0."""
    user_movie_matrix = ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)
    return user_movie_matrix

def converter_para_numpy(matrix):
    """Converte a matriz Pandas para um array NumPy."""
    return matrix.to_numpy()

# -------------------- Funções de Cálculo de Similaridade & Previsão --------------------

def calcular_similaridade(matriz):
    """(TODO) Calcula a similaridade entre usuários ou filmes."""
    # Exemplo: Usar similaridade do cosseno com scipy ou numpy
    pass  # Implementar futuramente

# -------------------- Funções de Recomendação & Visualização --------------------

def gerar_recomendacoes():
    """(TODO) Gera recomendações com base nos cálculos realizados."""
    pass  # Implementar futuramente

# -------------------- Função Principal para Execução do Código --------------------

def main():
    # Carregamento dos dados
    ratings, movies = carregar_dados()

    # Tratamento de valores nulos
    ratings = tratar_valores_nulos(ratings)
    movies = tratar_valores_nulos(movies)

    # Conversão do campo timestamp
    ratings = converter_timestamp(ratings)

    # Criação da matriz usuário-filme
    user_movie_matrix = criar_matriz_usuario_filme(ratings)
    print(user_movie_matrix.head())

    # Conversão para NumPy
    user_movie_array = converter_para_numpy(user_movie_matrix)
    print(f'Dimensões da matriz: {user_movie_array.shape}')

    # TODO: Implementar cálculo de similaridade e geração de recomendações

# Executa o programa
if __name__ == "__main__":
    main()
