from fastapi import FastAPI, HTTPException, Depends
import numpy as np
from typing import List
#from database import Base, SessionLocal
from schema import UserGet, PostGet, FeedGet
#from table_post import Post
#from table_user import User
#from table_feed import Feed
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List
import os
import pickle
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
import uvicorn
from datetime import datetime


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("catboost_model")
    # model = pickle.load(model_path) # пример как можно загружать модели

    from_file = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    model = from_file.load_model(model_path)

    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

#def get_db():
#    with SessionLocal() as db:
#        return db

def load_features() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM kravtsova_user_info').drop('index', axis=1)

def load_features_file() -> pd.DataFrame:
    return pd.read_csv('features.csv').drop('index', axis=1)

def load_posts() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM post_text_df')

def load_posts_file() -> pd.DataFrame:
    return pd.read_csv('df_posts.csv').drop('index', axis=1)

#загружаем таблицу с постами
model = load_models()
features = load_features()
#features.to_csv('features.csv')
#features = load_features_file()
df_posts = load_posts()
#df_posts.to_csv('df_posts.csv')
#df_posts = load_posts_file()
#копируем для будущего применения
df_post_original = df_posts.copy()

### обрабатываем таблицу с постами: удаляем текст и кодируем колонку topic (ohe)
one_hot_df = pd.get_dummies(df_posts['topic'], prefix='topic', drop_first=True, dtype=int)
df_posts = pd.concat((df_posts.drop(['topic', 'text'], axis=1), one_hot_df), axis=1)

catboost_features = ['user_id', 'age', 'city', 'exp_group', 'os_iOS', 'source_organic',
       'gender_1', 'country_Belarus', 'country_Cyprus', 'country_Estonia',
       'country_Finland', 'country_Kazakhstan', 'country_Latvia',
       'country_Russia', 'country_Switzerland', 'country_Turkey',
       'country_Ukraine', 'post_id', 'topic_covid', 'topic_entertainment',
       'topic_movie', 'topic_politics', 'topic_sport', 'topic_tech', 'year',
       'month', 'day', 'hour']





"""
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int,
		time: datetime = None,
		limit: int = 5) -> List[PostGet]:
     return user_features[user_features['user_id']==id]
"""

app = FastAPI()
@app.get("/post/recommendations/",  response_model=List[PostGet])
def recommended_posts(
		id: int,
		time: datetime = datetime.strptime('2021-12-01 00:00:00', '%Y-%m-%d  %H:%M:%S'),
		limit: int = 5) -> List[PostGet]:

    # обрабатываем время, в которое нам поступил запрос, передается вместе с запросом
    # date = datetime(year=2000, month=1, day=1, hour=1)
    global df_posts
    time_df = np.array([[time.year,
                         time.month,
                         time.day,
                         time.hour]
                        ])
    # делаем все в numpy для ускорения
    # дублируем строку с временем запроса для конкатенации со всеми постами
    time_df_repeated = np.repeat(time_df, df_posts.shape[0], axis=0)
    #отбираем информацию о нужном юзере
    user_info = np.array(features[features['user_id'] == id])
    # дублируем строку с информацией о юзере для конкатенации со всеми постами
    user_info_repeated = np.repeat(user_info, df_posts.shape[0], axis=0)
    # склеиваем
    #если не превратить его в нампай, то остается лишняя колонка с индексом
    df_full = np.concatenate((user_info_repeated, df_posts, time_df_repeated), axis=1)

    #df_full_pd = pd.DataFrame(df_full, columns=catboost_features)

    # используем модель для предсказания
    result = model.predict_proba(df_full)

    result = result[:, 1]  # берем вторую колонку с вероятностью таргета = 1
    indicies = result.argsort()  # сортируем массив и получаем индексы в порядке убывания вероятности

    # отбираем по индексам айди постов
    recommended_post_list = []
    for i in indicies[:5]:  # забираем 5 первых индексов
        recommended_post_list.append(df_post_original.iloc[i])

    final_result = []
    for i in recommended_post_list:
        post_obj = PostGet(id=i.iloc[0],
                           text=i.iloc[1],
                           topic=i.iloc[2]
                           )
        final_result.append(post_obj)
    print(final_result)
    return final_result


if __name__ == '__main__':
    uvicorn.run("app:app", reload=True, access_log=False)
