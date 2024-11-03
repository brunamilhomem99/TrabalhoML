import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


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
    # Usando a função cosine_similarity do scikit-learn
    similaridade = cosine_similarity(matriz)

    #  similaridade de cosseno manualmente com numpy usando a definição matemática: A.B / (||A||.||B||)
    #norm = np.linalg.norm(matriz, axis=1)  # Norma de cada vetor (usuário ou filme)
    #similaridade = np.dot(matriz, matriz.T) / (norm[:, None] * norm[None, :])

    return similaridade

# -------------------- Funções de Recomendação & Visualização --------------------

def gerar_recomendacoes(user_id, user_movie_matrix, similaridade, top_n=10):
    """(TODO) Gera recomendações com base nos cálculos realizados."""
    
    # Verifica se o user_id existe na matriz
    if user_id not in user_movie_matrix.index:
        raise ValueError(f"User ID {user_id} não encontrado na matriz.")

    user_index = user_movie_matrix.index.get_loc(user_id)  # Localiza o índice do usuário na matriz
    similar_users = similaridade[user_index]

    # Somamos os ratings ponderados pela similaridade com outros usuários
    recommendation_scores = user_movie_matrix.T.dot(similar_users) / np.array([np.abs(similar_users).sum()])
    
    # Identifica filmes que o usuário já avaliou
    user_rated = user_movie_matrix.iloc[user_index].nonzero()[0]
    
    # Remove as pontuações dos filmes já avaliados para evitar recomendá-los
    recommendation_scores[user_rated] = 0
    
    # Seleciona os top N filmes com maiores pontuações
    top_movies_indices = np.argsort(recommendation_scores)[-top_n:][::-1]
    recommended_movies = user_movie_matrix.columns[top_movies_indices]

    return recommended_movies.tolist()

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

    # Cálculo de similaridade
    similaridade = calcular_similaridade(user_movie_array)
    print("\nMatriz de Similaridade entre Usuários:\n", similaridade)

    # Geração de recomendações
    # Defina o ID do usuário para o qual você quer gerar recomendações
    user_id = 1  # Exemplo: gerar recomendações para o usuário com ID 1
    top_n = 5    # Número de recomendações desejadas

    # Geração de recomendações para o usuário especificado
    recomendacoes = gerar_recomendacoes(user_id, user_movie_matrix, similaridade, top_n=top_n)
    print(f"\nRecomendações para o usuário {user_id}:", recomendacoes)

    # Preparando os dados para visualização
    recomendados = pd.DataFrame({'movieId': recomendacoes})
    recomendados = recomendados.merge(movies, on='movieId', how='left')

    # Plotando as recomendações
    plt.figure(figsize=(10, 6))
    sns.barplot(x=recomendados['title'], y=recomendados.index)
    plt.title(f"Top {top_n} Recomendações para o Usuário {user_id}")
    plt.xlabel("Filme")
    plt.ylabel("Posição da Recomendação")
    plt.xticks(rotation=45)
    plt.show()


# Executa o programa
if __name__ == "__main__":
    main()
