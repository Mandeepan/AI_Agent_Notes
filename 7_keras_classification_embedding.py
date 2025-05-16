import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry
import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings

from sklearn.datasets import fetch_20newsgroups
import email
import re
import pandas as pd
import numpy as np
import keras
from keras import layers

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Add tqdm to Pandas
tqdmr.pandas()

# ...But suppress the experimental warning.
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


### STEP 0: DOWNLOAD THE OPEN SOURCE DATASET ###
newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# # uncomment the following to see the dataset first record
# print(newsgroups_train.target[0])
# print(newsgroups_train.data[0])

### STEP 1 : STRUCTURE THE DATA (FORMET IT) AND TAKE SAMPLES FROM CLEAN DATASET ###

def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate each entry to 5,000 characters
    text = text[:5000]

    return text


def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )
    # Clean up the text
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    # Match label to target name index
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])

    return df

def sample_data(df, num_samples, classes_to_keep):
    # Sample rows, selecting num_samples of each Label.
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)
    )

    df = df[df["Class Name"].str.contains(classes_to_keep)]

    # We have fewer categories now, so re-calibrate the label encoding.
    df["Class Name"] = df["Class Name"].astype("category")
    df["Encoded Label"] = df["Class Name"].cat.codes

    return df

df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

# # uncomment to see the top 5 row of the clean dataset
# print(df_train.head())

TRAIN_NUM_SAMPLES = 25
TEST_NUM_SAMPLES = 10
# Class name should contain 'sci' to keep science categories.
# Try different labels from the data - see newsgroups_train.target_names
CLASSES_TO_KEEP = "sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

# # uncomment to see the top 5 row of the sample dataset and counts
# print(df_train.head())
# print(df_train.value_counts("Class Name"))
# print(df_test.value_counts("Class Name"))


### STEP 2 : CREATE EMBEDDING WITH GEMINI WITHIN THE SAMPLE DATASETS ###
@retry.Retry(predicate=is_retriable, timeout=300.0)
def embed_fn(text: str) -> list[float]:
    # You will be performing classification, so set task_type accordingly.
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="classification",
        ),
    )

    return response.embeddings[0].values


def create_embeddings(df):
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df

df_train = create_embeddings(df_train)
df_test = create_embeddings(df_test)

# # uncomment to see the top 5 row of the sample dataset and newly created embeddings
# print(df_train.head())
# print(df_test.head())


### STEP 3 : BUILD THE CLASSIFICATION MODEL  ###
def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input([input_size], name="embedding_inputs"),
            layers.Dense(input_size, activation="relu", name="hidden"),
            layers.Dense(num_classes, activation="softmax", name="output_probs"),
        ]
    )

# Derive the embedding size from observing the data. The embedding size can also be specified
# with the `output_dimensionality` parameter to `embed_content` if you need to reduce it.
embedding_size = len(df_train["Embeddings"].iloc[0])

classifier = build_classification_model(
    embedding_size, len(df_train["Class Name"].unique())
)
print(classifier.summary())

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

### STEP 4 : TRAIN THE CLASSIFICATION MODEL AND EVALUAT IT ###
NUM_EPOCHS = 20
BATCH_SIZE = 32

# Split the x and y components of the train and validation subsets.
y_train = df_train["Encoded Label"]
x_train = np.stack(df_train["Embeddings"])
y_val = df_test["Encoded Label"]
x_val = np.stack(df_test["Embeddings"])

# Specify that it's OK to stop early if accuracy stabilises.
early_stop = keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)

# Train the model for the desired number of epochs.
history = classifier.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
)
# evaluting the model performance
print(classifier.evaluate(x=x_val, y=y_val, return_dict=True))


### STEP 4 : USE THE MODEL TO PREDICT/CLASSIFY ###
def make_prediction(text: str) -> list[float]:
    """Infer categories from the provided text."""
    # Remember that the model takes embeddings as input, so calculate them first.
    embedded = embed_fn(new_text)

    # And recall that the input must be batched, so here they are wrapped as a
    # list to provide a batch of 1.
    inp = np.array([embedded])

    # And un-batched here.
    [result] = classifier.predict(inp)
    return result

# This example avoids any space-specific terminology to see if the model avoids
# biases towards specific jargon.
new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""

result = make_prediction(new_text)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
    print(f"{category}: {result[idx] * 100:0.2f}%")

