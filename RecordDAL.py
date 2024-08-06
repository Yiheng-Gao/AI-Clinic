import mysql.connector

def save_case(question, answer, rate):
    try:
        # Establish the database connection
        mydb = mysql.connector.connect(
            host="localhost",
            user="",
            password="",
            database="aidoctor"
        )

        # Create a cursor object
        cursor = mydb.cursor()

        # Define the INSERT query
        query = "INSERT INTO records (question, answer, rate) VALUES (%s, %s, %s)"
        values = (question, answer, rate)

        # Execute the query
        cursor.execute(query,values)

        # Commit the transaction
        mydb.commit()

        print(f"Record inserted, ID: {cursor.lastrowid}")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # Close the cursor and connection
        cursor.close()
        mydb.close()

def get_records():
    try:
        # Establish the database connection
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="aidoctor"
        )

        # Create a cursor object
        cursor = mydb.cursor()

        # Define the SELECT query
        query = "SELECT question, answer FROM records"

        # Execute the query
        cursor.execute(query)

        # Fetch all the records
        records = cursor.fetchall()

        # Close the cursor and connection
        cursor.close()
        mydb.close()

        return records

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    

def get_full_records():
    try:
        # Establish the database connection
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="aidoctor"
        )

        # Create a cursor object
        cursor = mydb.cursor()

        # Define the SELECT query
        query = "SELECT question, answer, rate, fine_tune FROM records WHERE rate>=3"# AND fine_tune=0

        # Execute the query
        cursor.execute(query)

        # Fetch all the records
        records = cursor.fetchall()


        update_query = "UPDATE records SET fine_tune=1 WHERE rate>=3"
        cursor.execute(update_query)
        mydb.commit()



        # Close the cursor and connection
        cursor.close()
        mydb.close()

        return records

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

