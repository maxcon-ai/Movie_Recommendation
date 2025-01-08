import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer() 
cv  = CountVectorizer(max_features=5000,stop_words='english')

def convert(obj):
    lst = []
    for i in ast.literal_eval(obj):
        lst.append(i['name'])
    return lst

def convert_3(obj):
    lst = []
    l= 0
    
    for i in ast.literal_eval(obj):
        if l == 3:
            break
        lst.append(i['name'])
        l+=1
    return lst

def featch_director(obj):
    lst = []
    l= 0
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
           
            lst.append(i['name'])
            break
    return lst

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def recomende(movie):
    movie_index = r_movies[r_movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key= lambda x: x[1])[1:6]
    for i in  movie_list:
        print(r_movies.iloc[i[0]].title)

def suggest_movies(genres = None, min_ratting=None ,year = None):
    suggestion = f_movies
    # if genres:
    #     suggestion = suggestion[suggestion['genres'].apply(lambda x: any(g in x for g in genres))]
    if genres:
            suggestion = suggestion[suggestion['genres'].apply(lambda x: genres in x)]
    if year:
        suggestion = suggestion[suggestion['year']== int(year)]
    if min_ratting:
        suggestion = suggestion[suggestion['ratings']>=int(min_ratting)]
    else:
        suggestion = suggestion[suggestion['ratings']>=0]
    for i in list(suggestion['title'][:5]):
        print(i)

def chose1():
    name = input('Enter Movie Name:')
    print()
    print('Recommended Movie list:')
    print()
    recomende(name)

def chose2():
    name = input('Enter Movie Genres or Null:')
    year = input('Enter Movie year or Null:')
    ratting = input('Enter Movie year or Null:')
    print()
    print('Recommended Movie list:')
    print()

    suggest_movies(genres=name,min_ratting=ratting,year=year)

movies = pd.read_csv('movies.csv')
credits  = pd.read_csv('credits.csv')
movies= movies.merge(credits,on='title')
movies = movies[['id','title','overview','genres','keywords','crew','cast','release_date','vote_average']]
movies['year'] = pd.to_datetime(movies['release_date'],format='%d-%m-%Y').dt.year
movies = movies.drop(columns=['release_date'])
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_3)
movies['crew']= movies['crew'].apply(featch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['genres1'] = movies['genres']
movies['genres1'] = movies['genres1'].apply(lambda x:[i.replace(" ",'') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ",'') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ",'') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ",'') for i in x])
movies['tags']= movies['overview']+movies['genres1']+movies['keywords'] +movies['cast']+movies['crew'] 
r_movies = movies[['id','title','tags']]
r_movies['tags'] = r_movies['tags'].apply(lambda x:" ".join(x))
f_movies = movies[['id','title','genres','vote_average','year']]
f_movies = f_movies.rename(columns={'vote_average':'ratings'})
r_movies['tags'] = r_movies['tags'].apply(stem)
r_movies['tags'] = r_movies['tags'].apply(lambda x: x.lower())
vector = cv.fit_transform(r_movies['tags']).toarray()
similarity = cosine_similarity(vector)



while True:
    print()
    print()
    print('Movie Suggiestion Project:')
    print('Choice 1 for Suggiest By Name:')
    print('Choice 2 for Suggiest By Genres/Year/Ratings:')
    print('Choice 3 for Exit')
    print()
    x = input("Enter Your Chose:")
    if x == '3':
        break
    elif x == '2':
        chose2()
    elif x =='1':
        chose1()
    else:
        print('Value error not in choise ')