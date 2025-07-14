-- Initialize database for MLflow
CREATE DATABASE IF NOT EXISTS mlflow;

-- Create user for MLflow
CREATE USER IF NOT EXISTS 'mlflow'@'%' IDENTIFIED BY 'mlflow';
GRANT ALL PRIVILEGES ON mlflow.* TO 'mlflow'@'%';

-- Additional databases for the MLOps platform
CREATE DATABASE IF NOT EXISTS monitoring;
CREATE DATABASE IF NOT EXISTS experiments;

-- Grant privileges for monitoring
GRANT ALL PRIVILEGES ON monitoring.* TO 'mlflow'@'%';
GRANT ALL PRIVILEGES ON experiments.* TO 'mlflow'@'%';

FLUSH PRIVILEGES;