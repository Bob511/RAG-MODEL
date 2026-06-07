import mysql from "mysql2/promise";

const pool = mysql.createPool({
  host: "localhost",
  user: "root",
  password: "",
  database: "webspring",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

pool
  .getConnection()
  .then((connection) => {
    console.log("Connected to the database.");
    connection.release();
  })
  .catch((err) => {
    console.error("Error connecting to the database:", err);
  });

export default pool;
