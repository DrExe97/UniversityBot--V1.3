-- setup.sql
-- Run this ONCE to create the database and user
-- Command: psql -U postgres -f setup.sql

-- Create database
CREATE DATABASE university_chatbot1;

-- Connect to it
\c university_chatbot1;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Tables are auto-created by postgres_db.py on first run
-- This file just ensures the DB and extension exist

-- Optional: create a dedicated user (more secure than using postgres)
-- CREATE USER chatbot_user WITH PASSWORD 'your_secure_password';
-- GRANT ALL PRIVILEGES ON DATABASE university_chatbot TO chatbot_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chatbot_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chatbot_user;

SELECT 'Database setup complete!' AS status;
