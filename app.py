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
from loguru import logger


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


#эта функция загружает таблицы из файлов, если from_file=True (это удобно для отладки,
# для быстрого запуска сервиса), либо из базы данных, если from_file=False
def load_tables(from_file=False):
    if not from_file:
        logger.info("post features loading")
        post_features = batch_load_sql('SELECT * FROM kravtsova_post_features').drop('index', axis=1)
        # post_features.to_csv('post_features.csv')

        logger.info("user features loading")
        users = batch_load_sql('SELECT * from user_data')
        # users.to_csv('users.csv')

        logger.info("posts loading")
        df_posts = batch_load_sql('SELECT * FROM post_text_df')
        # df_posts.to_csv('df_posts.csv')

        logger.info("liked posts loading")
        df_liked = batch_load_sql("SELECT distinct post_id, user_id FROM public.feed_data where action='like'")
        # df_liked.to_csv('df_liked.csv')

    else:
        logger.info("post features loading")
        post_features = pd.read_csv('post_features.csv').drop('Unnamed: 0', axis=1)

        logger.info("user features loading")
        users = pd.read_csv('users.csv').drop('Unnamed: 0', axis=1)

        logger.info("posts loading")
        df_posts = pd.read_csv('df_posts.csv').drop('index', axis=1)

        logger.info("liked posts loading")
        df_liked = pd.read_csv('df_liked.csv').drop('Unnamed: 0', axis=1)

    return post_features, users, df_posts, df_liked


post_features, users, df_posts, df_liked = load_tables(from_file=False)
model = load_models()

# из этой таблицы будем доставать сами посты для ответа (по индексам)
#df_post_original = df_posts.copy()

catboost_features = [ 'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
                          'text', 'topic',
                          'Text_emb_1', 'Text_emb_2', 'Text_emb_3', 'Text_emb_4',
                         'Text_emb_5', 'Text_emb_6', 'Text_emb_7', 'Text_emb_8', 'Text_emb_9',
                         'Text_emb_10', 'Text_emb_11', 'Text_emb_12', 'Text_emb_13', 'Text_emb_14',
                         'Text_emb_15', 'Text_emb_16', 'Text_emb_17', 'Text_emb_18', 'Text_emb_19',
                         'Text_emb_20', 'Text_emb_21', 'Text_emb_22', 'Text_emb_23', 'Text_emb_24',
                         'Text_emb_25', 'Text_emb_26', 'Text_emb_27', 'Text_emb_28', 'Text_emb_29',
                         'Text_emb_30',
                              'month', 'day', 'hour']
col_list_required = ['age', 'city', 'exp_group', 'os', 'source', 'gender', 'country',
           'topic', 'month', 'day', 'hour', 'text', 'Text_emb_1', 'Text_emb_2',
           'Text_emb_3', 'Text_emb_4', 'Text_emb_5', 'Text_emb_6', 'Text_emb_7',
           'Text_emb_8', 'Text_emb_9', 'Text_emb_10', 'Text_emb_11', 'Text_emb_12',
           'Text_emb_13', 'Text_emb_14', 'Text_emb_15', 'Text_emb_16',
           'Text_emb_17', 'Text_emb_18', 'Text_emb_19', 'Text_emb_20',
           'Text_emb_21', 'Text_emb_22', 'Text_emb_23', 'Text_emb_24',
           'Text_emb_25', 'Text_emb_26', 'Text_emb_27', 'Text_emb_28',
           'Text_emb_29', 'Text_emb_30']



app = FastAPI()
logger.info("waiting for request")

@app.get("/post/recommendations/",  response_model=List[PostGet])
def recommended_posts(
		id: int,
		time: datetime = datetime.strptime('2021-12-01 00:00:00', '%Y-%m-%d  %H:%M:%S'),
		limit: int = 5) -> List[PostGet]:

    # обрабатываем время, в которое нам поступил запрос, передается вместе с запросом
    # date = datetime(year=2000, month=1, day=1, hour=1)
    global df_post_original, post_features

    post_features_ = post_features.copy() #эту таблицу будем отправлять в модель для предсказания

    time_df = np.array([[time.month,
                         time.day,
                         time.hour]
                        ])
    #сохраняем в список айди всех постов, которые данный пользователь уже лайкнул
    df_liked_list = list(df_liked[df_liked['user_id']==id]['post_id'])
    # дропаем все посты, которые пользователь уже лайкнул
    post_features_ = post_features_[~post_features_['post_id'].isin(df_liked_list)]
    # делаем все в numpy для ускорения
    # дублируем строку с временем запроса для конкатенации со всеми постами
    time_df_repeated = np.repeat(time_df, post_features_.shape[0], axis=0)
    #отбираем информацию о нужном юзере и дропаем user_id за ненадобностью
    user_info = np.array(users[users['user_id'] == id].drop('user_id', axis=1))
    # дублируем строку с информацией о юзере для конкатенации со всеми постами
    user_info_repeated = np.repeat(user_info, post_features_.shape[0], axis=0)
    # склеиваем
    #если не превратить его в нампай, то остается лишняя колонка с индексом
    df_full = np.concatenate((user_info_repeated, post_features_.drop('post_id', axis=1), time_df_repeated), axis=1)

    df_full_pd = pd.DataFrame(df_full, columns=catboost_features)

    # меняем порядок колонок на нужный нам
    df_full_pd = df_full_pd.reindex(col_list_required, axis="columns")

    # используем модель для предсказания
    result = model.predict_proba(df_full_pd)

    result = result[:, 1]  # берем вторую колонку с вероятностью таргета = 1
    indicies = np.flip(result.argsort())  # сортируем массив и получаем индексы в порядке убывания вероятности
    # отбираем по индексам айди постов
    final_result = []
    for i in indicies[:limit]:  # забираем первыe limit индексы и строки по индексам
        post_id = post_features_.iloc[i]['post_id'] #забираем по индексу post_id
        post = df_posts[df_posts['post_id']==post_id] #забираем пост по пост_айди
        #создаем объект класса PostGet из дф
        post_obj = PostGet(id=post['post_id'].iloc[0],
                           text=post['text'].iloc[0],
                           topic=post['topic'].iloc[0]
                           )
        final_result.append(post_obj)
    return final_result

