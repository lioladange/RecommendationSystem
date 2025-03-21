"""
Этот файл предназначен для обучения модели, и не заускаютсся во время работы сервиса
"""
# ### Изучаем таблицы feed_data, user_data, post_text_df, смотрим, какие есть колонки 

# In[50]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
from datetime import datetime


# In[ ]:


df = pd.read_sql("select * from feed_data limit 3",
           "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
           )
df.head(3)


# In[49]:


df = pd.read_sql("""select * 
                from user_data 
                limit 100 """,
           "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
           )
df.head(3)


# In[ ]:


df = pd.read_sql("select * from post_text_df limit 3",
           "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
           )
df


# ### Загружаем нужные нам данные и подготавливаем датасет к обучению

# In[51]:


### загружаем всю таблицу с постами

post_text_df = pd.read_sql("select * from post_text_df",
           "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
           )
np.array(df).shape


# In[ ]:


### подготавливаем таблицу с предобработанными фичами постов и загружаем ее в базу, делаем это все один раз, 
### далее внутри сервиса просто скачиваем таблицу с готовыми фичами 


#1. загружаем предобработанные эмбеддинги текстов из файла, делали в kaggle тк локально не хватает памяти
#reduced_embeddings = pd.DataFrame(np.genfromtxt('reduced_embeddings.csv', delimiter=","), columns=emb_col_names)

#2. добавляем пост_id к эмбеддингам, чтобы выгрузить таблицу в базу, и использовать уже предобработанные признаки
#post_text_df = pd.read_sql("select * from post_text_df",conn_url)
#post_features = pd.concat([post_text_df, reduced_embeddings], axis=1)

#3. добавляем длину текста
#post_features['text'] = post_features['text'].apply(lambda x: len(x))
#post_features.head()

#4. загружаем таблицу в бд
### загрузка таблицы с предобработанными данными о юзерах в базу

#post_features.to_sql("kravtsova_post_features",
#           "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml", if_exists='replace'
#           )

#df.to_sql('kravtsova_user_info', con=engine, if_exists='replace', index=True) # записываем таблицу

#далее просто используем эту готовую таблицу из базы

#5. Загружаем из базы
post_features = pd.read_sql("kravtsova_post_features",
           "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml").drop('index', axis=1)

post_features.head()


# In[18]:


### соединяем 2 таблицы (feed_data и user_data), загружаем и сохраняем по 15 записей на каждого юзера
#можно выгрузить базы или из файла df_for_learning_every_unic_user_15_post, раскомментировать нужное!!!

### сразу отметаем строки, где action=like, потому что они дублируют строки, где action=view и target=1 
#и удаляем колонку "action", тк она остается константная


### из-за неудовлетворительного качества модели, попробуем сделать так,
### чтобы каждый уникальный юзер попал в выборку

conn_url = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

query = """
with table_1 as(
    select user_id, post_id, target, timestamp,
    row_number() over (partition by user_id order by timestamp desc) as rn
    from feed_data
    where action != 'like')

select table_1.user_id, post_id, target, timestamp, gender,	age, country, city, exp_group, os, source
from table_1
join user_data as ud on table_1.user_id = ud.user_id
where rn<15"""

#загрузить из базы
#df = pd.read_sql(query,conn_url)

#записать в файл
#df.to_csv('df_for_learning_every_unic_user_15_post.csv')

#загрузить из файла
df = pd.read_csv('df_for_learning_every_unic_user_15_post.csv', parse_dates = ['timestamp']).drop(['Unnamed: 0'], axis=1)

df


# In[40]:


#смотрим, насколько сбалансированны классы
print(f"Доля положительного класса: {df['target'].mean()}")

#смотрим, сколько уникальных юзеров находится в выборке
print(f"Уникальных юзеров {df['user_id'].nunique()} из 163 тыс существующих")


# In[60]:


#работаем с таблицами

#преобразуем время в разные колонки: год, месяц, день, час
#df['year'] = df['timestamp'].apply(lambda x: x.year) #не используем год, потому что выборка неболшая и он константный, 2021
df['month'] = df['timestamp'].apply(lambda x: x.month)
df['day'] = df['timestamp'].apply(lambda x: x.day)
df['hour'] = df['timestamp'].apply(lambda x: x.hour)
#df = df.drop(['timestamp'], axis=1)  #пока оставляем, может пригодится, если будем разбивать на трейн и на тест по времени


#соединяем с датафреймом по post_id
df = df.merge(post_features, on='post_id')

#Избавляемся от айдишников постов и юзеров, тк они не участвуют в обучении модели
df = df.drop(['user_id', 'post_id'], axis=1)

#смотрим, что получилось
df.head(3)


# In[ ]:


### OHE и MTE  - пока не используем, в катбусте не нужно, тк в катбусте есть встроенная обработка кат. фичей. 
### Возможно пригодится для других моделей

#выбираем колонки для one_hot_encoding
#cols_for_ohe = ['os', 'source', 'gender', 'country', 'topic']

#выбираем колонки для mean_target_encoding
#cols_for_mte = ['city']


### делaем mte и ohe
#for col in cols_for_ohe:
#    one_hot_df = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
#    df = pd.concat((df.drop(col, axis=1), one_hot_df), axis=1) 

#for col in cols_for_mte:
#    mean_target = df.groupby(col)['target'].mean()
#    df[col] = df[col].map(mean_target)    
#    #для будущих предсказаний сохраняем значения city закодированные в файл 
#    mean_target.to_csv('city_mte.csv')


# In[61]:


#категориальные колонки
cat_cols = ['city', 'os', 'source', 'gender', 'country', 'topic', 'exp_group', 'month']


#порядок столбцов, как они идут в модель для предсказания + колонка target. 
#Нужно привести датайфрейм к такому виду
#Для удобства лучше отправлять их на обучение уже в таком виде, чтобы не изменять 
#потом порядок столбцов при отправке данных в модель

col_list_required = ['timestamp', 'age', 'city', 'exp_group', 'os',
'source', 'gender', 'country', 'topic', 'month', 'day', 'hour', 'text', 'target']


#создаем список имен для эмбеддингов текста
emb_col_names = []
for i in range(1,31):
    emb_col_names.append(f"Text_emb_{i}")


col_list_required = col_list_required + emb_col_names
print(col_list_required)

#сохраняем список колонок текущего датафрейма
col_list_old = list(df.columns)

#меняем порядок колонок на нужный нам
df = df.reindex(col_list_required, axis="columns")

#проверяем, что порядок изменился верно
print('Колонки имеют нужный порядок:', col_list_required == list(df.columns))


# In[62]:


df


# In[66]:


#Тут обычное разбиение на трейн и тест, без временной структуры.

from sklearn.model_selection import train_test_split
import numpy as np

y = df['target']
X = df.drop(['target', 'timestamp'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, 
                                                    random_state=1)

### теперь пробуем обучить катбуст
from catboost import CatBoostClassifier

#обучаем модель
catboost = CatBoostClassifier(iterations=300,
                              learning_rate=1,
                              depth=2,
                              random_seed=100)

catboost.fit(X_train, y_train, cat_cols, verbose=False) #в качестве третьего аргумента тут можно добавить категориальные колонки, 
#но тк мы вручную обработали категориальные колонки датасет, этого не требуется

print(f"Качество на трейне: {roc_auc_score(y_train, catboost.predict_proba(X_train)[:, 1])}")
print(f"Качество на тесте: {roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1])}")
#Качество на трейне: 0.7118261325171527
#Качество на тесте: 0.6971751235027213


# In[29]:


### Разбиение данных по времени

print(df['timestamp'].min())
print(df['timestamp'].max())
print("""Здесь мы видим, что у нас есть данные примерно за последние два месяца 21 года, поэтому колонку год можно дропнуть,
она констанстная. Можно тогда за отсечку трейна и теста взять 2021-12-14 00:00:00". Поссмотрим, какая доля данных получится 
в трейне, а какая в тесте:""")
print(f""" Train: {df[df['timestamp'] < datetime.strptime('2021-12-12 00:00:00', "%Y-%m-%d %H:%M:%S")].shape[0]/df.shape[0]}, 
 Test: {df[df['timestamp'] >= datetime.strptime('2021-12-12 00:00:00', "%Y-%m-%d %H:%M:%S")].shape[0]/df.shape[0]}""")
print("Получается около 20% данных идут в тест, остальные в трейн. Это нас устраивает.")

#Попробуем обучить на данных с временной структурой

### Разбивам на трейин и тест, используя отсечку '2021-12-12 00:00:00'
train_df = df[df['timestamp'] < datetime.strptime('2021-12-12 00:00:00', "%Y-%m-%d %H:%M:%S")]
test_df = df[df['timestamp'] >= datetime.strptime('2021-12-12 00:00:00', "%Y-%m-%d %H:%M:%S")]

X_train = train_df.drop(['target', 'timestamp'], axis=1)
X_test = test_df.drop(['target', 'timestamp'], axis=1)
y_train = train_df['target']
y_test = test_df['target']

### теперь пробуем обучить катбуст
from catboost import CatBoostClassifier

#обучаем модель
catboost = CatBoostClassifier(iterations=300,
                              learning_rate=1,
                              depth=2,
                              random_seed=100)

catboost.fit(X_train, y_train, cat_cols, verbose=False) #в качестве третьего аргумента тут можно добавить категориальные колонки, 
#но тк мы вручную обработали категориальные колонки датасет, этого не требуется

print(f"Качество на трейне: {roc_auc_score(y_train, catboost.predict_proba(X_train)[:, 1])}")
print(f"Качество на тесте: {roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1])}")
#Качество на трейне: 0.6706768417172458
#Качество на тесте: 0.6438821898398965


# In[78]:


### посмотрим на важность фичей

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

plot_feature_importance(catboost.feature_importances_,X_train.columns,'Catboost')


# In[ ]:


X_train.columns


# In[68]:


### Сохраним модель

catboost.save_model(
    'catboost_model_simple',
    format="cbm"
)


# In[ ]:


X_train


# In[ ]:




