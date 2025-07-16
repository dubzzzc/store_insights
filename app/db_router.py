import mysql.connector
import os

def get_store_connection(store_db, user, password):
    return mysql.connector.connect(
        host=os.getenv("CORE_DB_HOST"),
        user=user,
        password=password,
        database=store_db
    )
