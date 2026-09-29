from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), index=True)
    email = Column(String(200), unique=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)

    tickets = relationship("SupportTicket", back_populates="customer")


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)

    product_purchased = Column(String(200), nullable=True)
    date_of_purchase = Column(DateTime, nullable=True)

    ticket_type = Column(String(50), nullable=True)
    subject = Column(String(300), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(30), index=True)
    resolution = Column(Text, nullable=True)
    priority = Column(String(20), index=True)
    channel = Column(String(30), nullable=True)

    first_response_time = Column(String(50), nullable=True)
    time_to_resolution = Column(String(50), nullable=True)
    satisfaction_rating = Column(Float, nullable=True)

    customer = relationship("Customer", back_populates="tickets")