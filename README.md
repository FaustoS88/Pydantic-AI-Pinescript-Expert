# PineScript Expert

A Retrieval-Augmented Generation (RAG) AI Agent built with Pydantic AI that serves as an expert on Pine Script v6, TradingView's programming language for custom indicators and strategies. This agent leverages vector search technology and large language models to provide accurate, context-aware answers and generate working code examples.

## Features

- **Comprehensive Pine Script Knowledge**: Access the entire Pine Script v6 documentation through natural language queries
- **Code Generation**: Creates custom indicators and strategies based on user requirements
- **Interactive Interfaces**: Multiple ways to interact with the expert:
  - Web-based UI built with Streamlit
  - Interactive command-line interface
  - Single query execution for scripting
- **Multi-Provider Support**: Use either OpenAI or OpenRouter models as the LLM backend
- **Vector Search**: Utilizes pgvector for efficient semantic retrieval of relevant documentation
- **Full Documentation Processing**: Custom crawler that processes and analyzes TradingView's Pine Script documentation
- **Persistent Chat History**: Remember conversation context in the Streamlit UI
- **Extensive API**: Integrate with other systems using the Pydantic AI based architecture

## Screenshots

![Alt text](/Users/faustosaccoccio/Documents/crawl4ai-main/pydantic_ai_docs/Pydantic_AI_Pinescript_Expert/assets/asset2.png?raw=true "Streamlit-ui")
![Alt text](/Users/faustosaccoccio/Documents/crawl4ai-main/pydantic_ai_docs/Pydantic_AI_Pinescript_Expert/assets/asset1.png?raw=true "Streamlit-ui")

## Prerequisites

- **Python 3.9+**
- **PostgreSQL** with pgvector extension
- **OpenAI API key** (required for embeddings and default LLM)
- **OpenRouter API key** (optional, for alternative LLM providers)
- **Docker** (optional, for running PostgreSQL with pgvector)

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/pinescript-expert.git
   cd pinescript-expert
   ```

2. **Setup PostgreSQL with pgvector**

```bash
# Create a directory for Docker volume if it doesn't exist
mkdir -p ~/pinescript_postgres_data

# Run PostgreSQL with pgvector on port 54322 (different from standard 5432)
docker run --name pinescript-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -p 54322:5432 \
  -v ~/pinescript_postgres_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
   ```
   
3. **Install requirements and set up the environment**

   ```bash
   python setup.py
   # Edit the created .env file with your API keys
   ```

4. **Initialize the database**

   ```bash
   python init_db.py
   ```

5. **Populate the database with Pine Script documentation**

   ```bash
   python pinescript_crawler.py
   ```

6. **Start using the Streamlit UI**

   ```bash
   streamlit run streamlit_ui.py
   ```

   Or, for CLI interface:

   ```bash
   python interactive.py
   ```

## Usage Examples

### Web Interface

Start the Streamlit interface to interact with the agent through a web UI:

```bash
streamlit run streamlit_ui.py
```

### Command Line Interface

Launch an interactive shell for conversational access to the agent:

```bash
python interactive.py
```

Example session:
```
=================================================================
 Pine Script Expert Agent - Interactive Shell 
=================================================================
Ask any question about Pine Script v6 or type 'exit' to quit.
Type 'clear' to clear the conversation history.
=================================================================

> How do I create a simple moving average crossover strategy?

Processing your question...

================================================================================
To create a simple moving average crossover strategy in Pine Script v6, you'll need to:

1. Set up your indicator or strategy
2. Calculate two moving averages of different lengths
3. Determine crossover conditions
4. Generate buy/sell signals
5. Optionally add plotting for visualization

Here's a complete example:

```pine
//@version=6
strategy("Simple MA Crossover Strategy", overlay=true)

// Input parameters
fastLength = input.int(9, "Fast MA Length")
slowLength = input.int(21, "Slow MA Length")

// Calculate moving averages
fastMA = ta.sma(close, fastLength)
slowMA = ta.sma(close, slowLength)

// Determine crossover conditions
buySignal = ta.crossover(fastMA, slowMA)
sellSignal = ta.crossunder(fastMA, slowMA)

// Execute strategy
if (buySignal)
    strategy.entry("Buy", strategy.long)
    
if (sellSignal)
    strategy.entry("Sell", strategy.short)

// Plot moving averages
plot(fastMA, "Fast MA", color=color.blue)
plot(slowMA, "Slow MA", color=color.red)

// Plot buy/sell signals
plotshape(buySignal, "Buy Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(sellSignal, "Sell Signal", shape.triangledown, location.abovebar, color.red, size=size.small)
```

Key components explained:
- We use `ta.sma()` to calculate the simple moving averages
- `ta.crossover()` and `ta.crossunder()` detect when the fast MA crosses above or below the slow MA
- `strategy.entry()` executes buy and sell orders when crossovers occur
- `plot()` and `plotshape()` visualize the MAs and signals on the chart

You can customize this by changing:
- MA types (SMA, EMA, WMA, etc.)
- Length parameters
- Adding additional conditions
- Adding stop-loss and take-profit levels
================================================================================

> 
```

### Single Query Mode

Use the agent for a one-time query:

```bash
python run.py query "How do I calculate RSI in Pine Script?"
```

### Database Inspection

The project includes tools to inspect the vector database:

```bash
# Count documents in the database
python db_inspect.py count

# List document titles (first 20)
python db_inspect.py list

# View a specific document
python db_inspect.py view 1

# Test search functionality
python db_inspect.py search "how to use request.security for different timeframes"
```

## Key Components

- **`agent.py`**: Core agent implementation with RAG capabilities
- **`pinescript_crawler.py`**: Documentation crawler and vector database population
- **`db_schema.py`**: Database schema definitions
- **`streamlit_ui.py`**: Web-based user interface
- **`streamlit_persistent.py`**: Enhanced UI with persistent chat history
- **`interactive.py`**: Command-line interface
- **`run.py`**: Convenience runner for various operation modes
- **`init_db.py`**: Database initialization
- **`clear_database.py`**: Database cleaning utility
- **`db_inspect.py`**: Database inspection tools

## Advanced Configuration

### Model Settings

You can customize the agent's model settings in `agent.py`:

```python
pinescript_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=Dependencies,
    result_type=PineScriptResult,
    system_prompt=(
        "You are a Pine Script v6 expert assistant..."
    ),
    model_settings={
        "temperature": 0.2,  # Adjust for more/less creativity
        "max_tokens": 2000   # Adjust for longer/shorter responses
    }
)


# Create a custom model that uses OpenRouter
    class OpenRouterModel(OpenAIModel):
        def __init__(self, model_name="deepseek/deepseek-chat"): # Change model here
            super().__init__(
                model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )
    
    # Use a model ID that OpenRouter actually supports
    return OpenRouterModel("deepseek/deepseek-chat") # Change model here
```

### Using OpenRouter

For access to alternative LLM providers:

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get an API key
3. Add it to your `.env` file as `OPENROUTER_API_KEY`

The agent will automatically use OpenRouter models when the API key is present.

### Custom Database Connection

Configure database settings in the `.env` file:

```
DATABASE_URL=postgresql://username:password@hostname:port/database
```
## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Pydantic AI](https://ai.pydantic.dev/) for the agent framework
- [TradingView](https://www.tradingview.com/) for the Pine Script language and documentation
- [OpenAI](https://openai.com/) and [OpenRouter](https://openrouter.ai/) for LLM capabilities
- [pgvector](https://github.com/pgvector/pgvector) for vector search functionality
- [Streamlit](https://streamlit.io/) for the web interface
