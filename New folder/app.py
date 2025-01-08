from flask import Flask, jsonify, request, render_template
import requests
import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


TMDB_API_KEY = '065dcb4dfc03b3eb95f8f9d2996a6ab8'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')


movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'crew', 'cast', 'release_date', 'vote_average']]
movies['year'] = pd.to_datetime(movies['release_date'], format='%d-%m-%Y').dt.year
movies.drop(columns=['release_date'], inplace=True)
movies.dropna(inplace=True)
movies['year'] = movies['year'].astype(int)

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert_3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []
def stem(text):
    y=[ps.stem(i) for i in text.split()]
    # for i in text.split():
    #     y.append(ps.stem(i))
    return " ".join(y)
def get_movie_thumbnail(movie_id):
    url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path', None)
        if poster_path:
            return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    return None

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = (movies['overview'] + movies['genres'] + movies['keywords'] +
                  movies['cast'] + movies['crew']).apply(lambda x: " ".join(x).lower())
r_movies = movies[['id', 'title', 'tags']]
vector = cv.fit_transform(r_movies['tags'].apply(stem)).toarray()
similarity = cosine_similarity(vector)
f_movies = movies[['id', 'title', 'genres', 'vote_average', 'year']].rename(columns={'vote_average': 'ratings'})

def suggest_movies(genres=None, min_rating=None, year=None):
    suggestion = f_movies
    if genres:
        suggestion = suggestion[suggestion['genres'].apply(lambda x: genres in x)]
    if year:
        suggestion = suggestion[suggestion['year'] == int(year)]
    if min_rating :
        suggestion = suggestion[suggestion['ratings'] >= float(min_rating)]
    else:
        suggestion = suggestion[suggestion['ratings'] >= 0]
    return suggestion[['title', 'ratings', 'year']].to_dict(orient='records')[:5]

def recommend(movie):
    movie_index = r_movies[r_movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in movie_list[:6]:
        title = r_movies.iloc[i[0]].title
        movie_id = r_movies.iloc[i[0]].id
        thumbnail = get_movie_thumbnail(movie_id)
        recommendations.append({'title': title, 'thumbnail': thumbnail})
    return recommendations
def recommend_story(storyline):
    input_vector = cv.transform([storyline]).toarray()
    distances = cosine_similarity(input_vector, vector)
    movie_list = sorted(list(enumerate(distances[0])), key=lambda x: x[1], reverse=True)[:10]

    # Fetch recommended movie titles
    return [movies.iloc[i[0]].title for i in movie_list]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_name = request.form.get('movie_name')
    if not movie_name:
        return render_template('index.html', message="Please enter a movie name.")
    try:
        recommendations = recommend(movie_name)
        return render_template('recommend.html',search =recommendations[0] , recommendations=recommendations[1:], movie_name=movie_name)
    except:
        return render_template('index.html', message="Movie not found in the dataset.")
@app.route('/story', methods=['POST'])
def recommend_movies_by_story():
    story_line = request.form.get('story')
    if not story_line:
        return render_template('index.html', message="Please enter a story line.")
    try:
        recommendations = recommend_story(story_line)
        return render_template('recommendStory.html', recommendations=recommendations)
    except:
        return render_template('index.html', message="Movie not found in the dataset.")

@app.route('/filter', methods=['POST'])
def filter_movies():
    genres = request.form.get('genres', None)
    min_rating = request.form.get('min_rating', None)
    year = request.form.get('year', None)
    suggestions = suggest_movies(genres, min_rating, year)
    return render_template('filter.html', suggestions=suggestions)
    
@app.route('/search', methods=['GET'])
def search_movies():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    suggestions = movies[movies['title'].str.contains(query, case=False) | 
                         movies['cast'].apply(lambda x: any(query in name.lower() for name in x))]
    return jsonify(suggestions['title'].tolist()[:10])


if __name__ == "__main__":
    app.run(debug=True)
