const db = require('./db');
const bcrypt = require('bcryptjs');

async function initialize() {
    try {
        // 1. Create Tables
        await db.query(`
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            first_name VARCHAR(50) NOT NULL,
                                          last_name VARCHAR(50) NOT NULL,
                                          email VARCHAR(255) UNIQUE NOT NULL,
                                          password_hash TEXT NOT NULL,
                                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reports (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                                            report_type VARCHAR(50) NOT NULL,
                                            report_data JSONB NOT NULL,
                                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        `);

        // 2. Insert Admin User if they don't exist
        const adminEmail = 'admin@admin.com';
        const check = await db.query('SELECT id FROM users WHERE email = $1', [adminEmail]);

        if (check.rows.length === 0) {
            const password_hash = await bcrypt.hash('adminpassword123', 10);
            await db.query(
                'INSERT INTO users (first_name, last_name, email, password_hash) VALUES ($1, $2, $3, $4)',
                           ['Admin', 'User', adminEmail, password_hash]
            );
            console.log('Admin user created: admin@admin.com / adminpassword123');
        }
    } catch (err) {
        console.error('Initialization error:', err);
    }
}

/*
Getting Started
Setup Postgres: Ensure PostgreSQL is running locally, install it in your server file. Create a database:
createdb gp_database

Environment Variables: Copy .env.example to .env and update the DATABASE_URL with your local PostgreSQL credentials that are in the .env file.*/

initialize();
