import os
import io
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry
from pprint import pprint

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define a retry policy. The model might make multiple consecutive calls automatically
# for a complex query, this ensures the client retries if it hits quota limits.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

### 1. SERACH WITHOUT AND WITH THE SEARCH GROUNDING TOOL( WITH ACCESS TO UPDATED SEARCH INFO FROM GOOGLE) ###
# Ask for information without search grounding.

question = "What were the S&P closing prices in the previous 5 trading days?"

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=question)
print("\n>>>>>>>>>>>>>>No Search Grounding Result: \n")
print(response.text)

# And now re-run the same query with search grounding enabled.
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

def query_with_grounding():
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=question,
        config=config_with_search,
    )
    return response.candidates[0]


rc = query_with_grounding()
print("\n>>>>>>>>>>>>>>With Search Grounding Result: \n")
print(rc.content.parts[0].text)

### 2. RESPONSE COMES WITH THE METADATA ###
# # uncomment the following
# print("\n>>>>>>>>>>>>>>Grounding Chunks: \n")
# while not rc.grounding_metadata.grounding_supports or not rc.grounding_metadata.grounding_chunks:
#     # If incomplete grounding data was returned, retry.
#     rc = query_with_grounding()

# chunks = rc.grounding_metadata.grounding_chunks
# for chunk in chunks:
#     print(f'{chunk.web.title}: {chunk.web.uri}')


# print("\n>>>>>>>>>>>>>>Search Entry Point: \n")
# print(rc.grounding_metadata.search_entry_point.rendered_content)


# print("\n>>>>>>>>>>>>>>Grounding Supports: \n")
# supports = rc.grounding_metadata.grounding_supports
# for support in supports:
#     pprint(support.to_json_dict())

# # can further format the response using the grounding support tool
# print("\n>>>>>>>>>>>>>>Formated Responses with Grounding Supports: \n")
# markdown_buffer = io.StringIO()

# # Print the text with footnote markers.
# markdown_buffer.write("Supported text:\n\n")
# for support in supports:
#     markdown_buffer.write(" * ")
#     markdown_buffer.write(
#         rc.content.parts[0].text[support.segment.start_index : support.segment.end_index]
#     )

#     for i in support.grounding_chunk_indices:
#         chunk = chunks[i].web
#         markdown_buffer.write(f"<sup>[{i+1}]</sup>")

#     markdown_buffer.write("\n\n")

# # And print the footnotes.
# markdown_buffer.write("Citations:\n\n")
# for i, chunk in enumerate(chunks, start=1):
#     markdown_buffer.write(f"{i}. [{chunk.web.title}]({chunk.web.uri})\n")


# print(markdown_buffer.getvalue())


### 3. ADD TASK (DRAW GRPAH IN THIS EXAMPLE) WITH THE UPDATED INFO

def show_response(response):
    os.makedirs("outputs", exist_ok=True)
    image_counter = 1

    for p in response.candidates[0].content.parts:
        if p.text:
            print(p.text)
        elif p.inline_data:
            mime_type = p.inline_data.mime_type
            extension = mime_type.split('/')[-1] if '/' in mime_type else 'bin'
            filename = f"outputs/11_updated_info_graph_{image_counter}.{extension}"
            
            # Directly write binary data without decoding
            with open(filename, "wb") as img_file:
                img_file.write(p.inline_data.data)
            print(f"Image saved directly to {filename}")
            image_counter += 1
        else:
            print(p.to_json_dict())


config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0,
)

chat = client.chats.create(model='gemini-2.0-flash')

# step 1, get prices info
response = chat.send_message(
    message="What were the S&P500 close prices in the previous 7 days?",
    config=config_with_search,
)
show_response(response)

# step 2, draw graph
config_with_code = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    temperature=0.0,
)

response = chat.send_message(
    message="Now plot this as a seaborn chart. Show the average price line too.",
    config=config_with_code,
)

show_response(response)