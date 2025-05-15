import os
from dotenv import load_dotenv
from google import genai 
from google.genai import types

import pandas as pd
import seaborn as sns

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

### STEP 0: PREPARING MODEL CONFIG and DATA for vector database ###

# show the current available models that support the embedContent function
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)
print("==============================")

### STEP 1: USER SEMANTIC SIMILARITY TASK TYPE TO EMBED THE INPUT TEXTS ###
texts = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick rbown fox jumps over the lazy dog.',
    'teh fast fox jumps over the slow woofer.',
    'a quick brown fox jmps over lazy dog.',
    'brown fox jumping over dog',
    'fox > dog',
    # Alternative pangram for comparison:
    'The five boxing wizards jump quickly.',
    # Unrelated text, also for comparison:
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.',
]


response = client.models.embed_content(
    model='models/text-embedding-004',
    contents = texts,
    config = types.EmbedContentConfig(task_type="semantic_similarity")
)

### STEP 2: Output the similarity of each of the text in texts variable towards to others in texts ###
# for better visuability, keep each text within 30 charaters
def truncate(t: str, limit: int = 50) -> str:
  """Truncate labels to fit on the chart."""
  if len(t) > limit:
    return t[:limit-3] + '...'
  else:
    return t

truncated_texts = [truncate(t) for t in texts]



# Set up the embeddings in a dataframe.
df = pd.DataFrame([e.values for e in response.embeddings], index=truncated_texts)
# Perform the similarity calculation
sim = df @ df.T
# Draw!
print(sns.heatmap(sim, vmin=0, vmax=1, cmap="Greens"))


### STEP 3: input an existing text and show the similarity score of each record in texts ###
score_result= sim['The quick brown fox jumps over the lazy dog.'].sort_values(ascending=False)
print(score_result)