import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry
import sqlite3
import textwrap
from pprint import pformat
import asyncio

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define a retry policy. The model might make multiple consecutive calls automatically
# for a complex query, this ensures the client retries if it hits quota limits.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

### STEP 0: SET UP DATABASED IF HAVENT ###
# If the file doesn't exist, it will be created in your project folder
db_conn = sqlite3.connect("sample.db")
# Create a cursor object to interact with the DB
cursor = db_conn.cursor()

# read the set up sql query to create table and insert a few records ( if not pre-exist yet)
# Read and execute the setup SQL file
with open("./src/database_setup_script_8.sql", "r") as f:
    database_sql_script = f.read()
cursor.executescript(database_sql_script)  # Executes multiple SQL statements
# Commit changes and close the connection
db_conn.commit()
db_conn.close()

### STEP 1: GET DATABASE INFO ( TABLE NAMES, COLUMN NAMES ETC) THEN FETCH DATA ###
def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    # Include print logging statements so you can see when functions are being called.
    print(' - DB CALL: list_tables()')

    cursor = db_conn.cursor()

    # Fetch the table names.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    return [t[0] for t in tables]


def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(f' - DB CALL: describe_table({table_name})')

    cursor = db_conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name});")

    schema = cursor.fetchall()
    # [column index, column name, column type, ...]
    return [(col[1], col[2]) for col in schema]

def execute_query(sql: str) -> list[list[str]]:
    """Execute an SQL statement, returning the results."""
    print(f' - DB CALL: execute_query({sql})')

    cursor = db_conn.cursor()

    cursor.execute(sql)
    return cursor.fetchall()

db_file = "sample.db"
db_conn = sqlite3.connect(db_file)
# print("\n >>>>>>>Tables In This Database >>>>>>> \n")
# print(list_tables())
# print("\n >>>>>>>Columns In A Specific Table >>>>>>> \n")
# print(describe_table("products"))
# print(describe_table("orders"))
# print(describe_table("staff"))
# sql_query="select * from products"
# print("\n >>>>>>>Executing SQL Query : %s >>>>>>> \n"%sql_query)
# print(execute_query(sql_query))

### STEP 2 : IMPLEMENTING FUNCTION CALLS WITHOUT LIVE API ( PRINT CHAT HISTORY INSTEAD OF SHOWING LIVE RESPONSES) ###
db_tools = [list_tables, describe_table, execute_query]

instruction = """You are a helpful chatbot that can interact with an SQL database
for a computer store. You will take the users questions and turn them into SQL
queries using the tools available. Once you have the information you need, you will
answer the user's question using the data returned.

Use list_tables to see what tables are present, describe_table to understand the
schema, and execute_query to issue an SQL SELECT query."""

client = genai.Client(api_key=GOOGLE_API_KEY)

# Start a chat with automatic function calling enabled.
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=instruction,
        tools=db_tools,
    ),
)
resp = chat.send_message("What is the most expensive product?")
# print(f"\n{resp.text}")
resp1 = chat.send_message("What is the average price among all products?")
# print(f"\n{resp1.text}")
resp2 = chat.send_message("Who is the salesperson with the largest total dollar amount of all his orders?")
# print(f"\n{resp2.text}")
response = chat.send_message('What products should salesperson Alice focus on to round out her portfolio? Explain why.')
# print(f"\n{response.text}")

def print_chat_turns(chat):
    """Prints out each turn in the chat history, including function calls and responses."""
    for event in chat.get_history():
        print(f"{event.role.capitalize()}:")

        for part in event.parts:
            if txt := part.text:
                print(f'  "{txt}"')
            elif fn := part.function_call:
                args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                print(f"  Function call: {fn.name}({args})")
            elif resp := part.function_response:
                print("  Function response:")
                print(textwrap.indent(str(resp.response['result']), "    "))

        print()
# # uncommend to see the chat history result
# print_chat_turns(chat)

### STEP 3 : IMPLEMENTING FUNCTION CALLS WITH LIVE API  ###
async def handle_response(stream, tool_impl=None):
  """Stream output and handle any tool calls during the session."""
  all_responses = []

  async for msg in stream.receive():
    all_responses.append(msg)

    if text := msg.text:
      # Output any text chunks that are streamed back.
      if len(all_responses) < 2 or not all_responses[-2].text:
        # Display a header if this is the first text chunk.
        print('### Text \n')

      print(text, end='')

    elif tool_call := msg.tool_call:
      # Handle tool-call requests.
      for fc in tool_call.function_calls:
        print('### Tool call \n')

        # Execute the tool and collect the result to return to the model.
        if callable(tool_impl):
          try:
            result = tool_impl(**fc.args)
          except Exception as e:
            result = str(e)
        else:
          result = 'ok'

        tool_response = types.LiveClientToolResponse(
            function_responses=[types.FunctionResponse(
                name=fc.name,
                id=fc.id,
                response={'result': result},
            )]
        )
        await stream.send(input=tool_response)

    elif msg.server_content and msg.server_content.model_turn:
      # Print any messages showing code the model generated and ran.

      for part in msg.server_content.model_turn.parts:
          if code := part.executable_code:
            print(
                f'### Code\n```\n{code.code}\n```')

          elif result := part.code_execution_result:
            print(f'### Result: {result.outcome}\n'
                             f'```\n{pformat(result.output)}\n```')

          elif img := part.inline_data:
            print(img.data)

  print()
  return all_responses

model = 'gemini-2.0-flash-exp'
live_client = genai.Client(api_key=GOOGLE_API_KEY,
                           http_options=types.HttpOptions(api_version='v1alpha'))

# Wrap the existing execute_query tool you used in the earlier example.
execute_query_tool_def = types.FunctionDeclaration.from_callable(
    client=live_client, callable=execute_query)

# Provide the model with enough information to use the tool, such as describing
# the database so it understands which SQL syntax to use.
sys_int = """You are a database interface. Use the `execute_query` function
to answer the users questions by looking up information in the database,
running any necessary queries and responding to the user.

You need to look up table schema using sqlite3 syntax SQL, then once an
answer is found be sure to tell the user. If the user is requesting an
action, you must also execute the actions.
"""

config = {
    "response_modalities": ["TEXT"],
    "system_instruction": {"parts": [{"text": sys_int}]},
    "tools": [
        {"code_execution": {}},
        {"function_declarations": [execute_query_tool_def.to_json_dict()]},
    ],
}

async def insert_records():
    async with live_client.aio.live.connect(model=model, config=config) as session:
        message = "Please generate and insert 5 new rows in the orders table."
        print(f"> {message}\n")

        await session.send(input=message, end_of_turn=True)
        await handle_response(session, tool_impl=execute_query)

# asyncio.run(insert_records())

async def generate_plot():
    async with live_client.aio.live.connect(model=model, config=config) as session:
        message = "Can you figure out the number of orders that were made by each of the staff?"

        print(f"> {message}\n")
        await session.send(input=message, end_of_turn=True)
        await handle_response(session, tool_impl=execute_query)

        message = "Generate and run some code to plot this as a python seaborn chart"

        print(f"> {message}\n")
        await session.send(input=message, end_of_turn=True)
        await handle_response(session, tool_impl=execute_query)

asyncio.run(generate_plot())