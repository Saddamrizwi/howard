# from sqlalchemy import Boolean, Column, ForeignKey, Integer, String,DateTime,Text
# from sqlalchemy.orm import relationship
# from datetime import datetime
# # from database import Base

# class User(Base):
#     __tablename__ = 'users'

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String)
#     email = Column(String)
#     password = Column(String)
#     is_active = Column(Boolean, default=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow)
#     chat_history = relationship('ChatHistory', back_populates='user')



# class ChatHistory(Base):
#     __tablename__ = 'chat_history'

#     chat_id = Column(Integer, primary_key=True)
#     # user_id = Column(Integer, ForeignKey('users.id'))
#     email_id=Column(Integer, ForeignKey('users.email'))
#     all_messages = Column(Text)  
#     favourite = Column(Boolean)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     topic = Column(Text) 
#     user = relationship('User', back_populates='chat_history')
