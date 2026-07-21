const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});

// Test the connection on startup
pool.query('SELECT NOW()', (err, res) => {
    if (err) console.error('Database connection error:', err.stack);
    else console.log('Connected to PostgreSQL successfully at:', res.rows[0].now);
});

module.exports = {
    query: (text, params) => pool.query(text, params),
};
