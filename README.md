# genai
Repository for the agentic ai and multi agent orchestration projects



docker ps
docker --version --> checking the docker version
docker run -d --name pgvector-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=ragdb -p 5432:5432 ankane/pgvector --> running the pgvector db of postgres from ankane/pgvector docker image

CREATE EXTENSION IS NOT EXISTS vector; --> run inside the db query to enable vector extension

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)
); -- creating a TABLE embeddings with text and the embedding vector

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE youtube_chunks (
    id SERIAL PRIMARY KEY,
    video_id TEXT,
    chunk_text TEXT,
    start_time FLOAT,
    end_time FLOAT,
    duration FLOAT,
    token_count INT,
    embedding VECTOR(1536)
);

docker ps --> from the command o/p check if the postgres db is running on the port 5432
docker exec -it pgvector-db psql -U postgres -d ragdb --> enter inside the container to run the db query

#\l --> to list all the databases
#\dt --> to list all the tables
#\c {database_name} --> to switch to another db
