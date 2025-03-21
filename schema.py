import datetime
from pydantic import BaseModel


class UserGet(BaseModel):
    id : int
    gender : int
    age : int
    country : str
    city : str
    exp_group : int
    os : str
    source : str
    class Config:
        from_attributes = True

class PostGet(BaseModel):
    id : int
    text : str
    topic : str
    class Config:
        from_attributes = True


class FeedGet(BaseModel):
    #id : int
    user_id : int
    post_id : int
    action : str
    time : datetime.datetime
    user: UserGet
    post: PostGet
    class Config:
        from_attributes = True