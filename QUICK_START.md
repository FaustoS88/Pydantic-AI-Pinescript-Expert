# Quick Start Guide

This guide provides quick step-by-step instructions to get the PineScript Expert Agent up and running.

## 1. Set Up Environment

Make sure you have Python 3.9+ installed, then:

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
cp .env.example .env
# Edit .env with your API keys
```

## 2. Set Up Database

```bash
# Start PostgreSQL with pgvector using Docker
docker run --name pinescript-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 54322:5432 \
  -v ~/pinescript_postgres_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16

# Initialize the database
python init_db.py
```

## 3. Crawl Documentation & Populate Database

```bash
# Run the crawler to get PineScript documentation
python pinescript_crawler.py
```

## 4. Use the Agent

```bash
# Run interactive mode
python run.py interactive

# Or run a single query
python run.py query "How do I create a moving average in Pine Script?"
```

## Checking Database Status

```bash
# Check database setup
python db_inspect.py count

# View sample documents
python db_inspect.py list

# Test search functionality
python db_inspect.py search "moving average crossover"
```

## Troubleshooting

1. **Docker Issues**: Make sure Docker is running and port 54322 is available
2. **Database Connection**: Verify connection string in `.env` file
3. **Missing pgvector**: Run `python init_db.py` to check if pgvector is correctly installed
4. **Embedding Issues**: Make sure your OpenAI API key is valid and has sufficient quota, you can test it with `python api_debug.py`

For more detailed instructions, see the full [README.md](README.md) and [scripts.md](scripts.md).
