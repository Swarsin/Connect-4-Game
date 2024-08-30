import sqlite3 as sql

def GetList():
    try:
        connect = sql.connect("c4db.db")
        cursor = connect.cursor()

        # cursor.execute("""
        #     CREATE TABLE user (
        #                username TEXT PRIMARY KEY,
        #                wins INTEGER
        #     )
        #             """)

        cursor.execute("""
                    SELECT username, wins 
                    FROM users
                    """)

        rows = cursor.fetchall()
        list = [(row[0], row[1]) for row in rows]
        return list
    except Exception as error:
        print(error)
    finally:
        connect.close()

print(GetList())
