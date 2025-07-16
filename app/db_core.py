import mysql.connector
import os

def get_core_connection():
    return mysql.connector.connect(
        host=os.getenv("CORE_DB_HOST"),
        user=os.getenv("CORE_DB_USER"),
        password=os.getenv("CORE_DB_PASSWORD"),
        database=os.getenv("CORE_DB_NAME")
    )
