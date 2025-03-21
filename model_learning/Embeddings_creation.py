#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install transformers
#!pip install torch
#!pip install sentence_transformers
#!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# In[1]:


from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[17]:


### код ниже работает в kaggle, но не работает тут, не хватает памяти mps. device надо заменить на сuda.
# поэтому готовый файл с эмбеддингами просто загружаем


"""
import gc
!export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

device = torch.device("cuda") 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    i = 0
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        i += 1
        chunk_dataframe = list(chunk_dataframe['text'])
        
        with torch.no_grad():
            inputs = tokenizer(chunk_dataframe,
                               return_tensors="pt",
                               padding=True,
                               truncation=True).to(device)
            
            outputs = model(**inputs)
            del inputs
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            print(batch_embeddings.shape)
            del outputs

        if i==1:
            embeddings = batch_embeddings.cpu().numpy()
        else:
            embeddings = np.concatenate([embeddings, batch_embeddings.cpu().numpy()])
        del batch_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Succesful iteration number {i}, len(emb): {len(embeddings)}")
    conn.close()
    
    #return pd.concat(embeddings, ignore_index=True)
    return embeddings

def load_text() -> pd.DataFrame:
    return batch_load_sql('SELECT text from post_text_df')

emb = load_text()
"""


# In[45]:


#достаем массив из файла
emb = np.genfromtxt('raw_text_embeddings.csv', delimiter=",")

#в этом массиве всего три пропуска
emb_nan = emb[np.isnan(emb)]
print(emb_nan.shape)

#поэтому заполним пропущенные значения средним по массиву
emb[np.isnan(emb)] = np.nanmean(emb)

#проверяем, что пропущенных значений не осталось
emb_nan = emb[np.isnan(emb)]
print(emb_nan.shape)

# 3. Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=30)
reduced_embeddings = pca.fit_transform(emb)


# 5. Check the shape of the reduced embeddings
print(reduced_embeddings.shape)  # Should be (number_of_sentences, 2)


#сохраняем полученные компоненты в файл
np.savetxt("reduced_embeddings.csv", reduced_embeddings, delimiter=",")

