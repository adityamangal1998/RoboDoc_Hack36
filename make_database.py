import sqlite3

#database maker
conn = sqlite3.connect('database.db')
print ("Opened database successfully")

conn.execute('CREATE TABLE users (userid TEXT, pass TEXT)')
print ("Table created successfully");
conn.close()

mydb = sqlite3.connect(

  database="database.db"
)
#database fetcher
mycursor = mydb.cursor()

mycursor.execute("SELECT userid, pass FROM users")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)