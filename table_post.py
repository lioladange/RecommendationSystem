from database import Base, SessionLocal, engine
from sqlalchemy import Column, Integer, Text, String
from sqlalchemy.orm import relationship

class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    topic = Column(String)


if __name__ == '__main__':
    session = SessionLocal()
    results = session.query(Post).filter(Post.topic == "business").order_by(Post.id.desc()).limit(10).all()

    result_list = []
    for x in results:
        result_list.append(x.id)

    print(result_list)