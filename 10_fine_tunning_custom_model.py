import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry

from sklearn.datasets import fetch_20newsgroups
import email
import re
import pandas as pd
import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings

from collections.abc import Iterable
import random
import datetime
import time

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

### STEP 0: FIND MODELS THAT SUPPORT CREATETURNEDMODEL AN DOWNLOAD DATASET ###

for model in client.models.list():
    if "createTunedModel" in model.supported_actions:
        print(model.name)
# output in May 2025 : models/gemini-1.5-flash-001-tuning

newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# #uncomment to set the sample record from the dataset downloaded
# print(newsgroups_train.data[0])

### STEP 1: CLEASE DATASET AND STRUCTURE IT ###
def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate the text to fit within the input limits
    text = text[:40000]

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
    df["Class Name"] = df["Class Name"].astype("category")

    return df

df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)
# #uncomment to see the dataset head
# print(df_train.head())
TRAIN_NUM_SAMPLES = 50
TEST_NUM_SAMPLES = 10
# Keep rec.* and sci.*
CLASSES_TO_KEEP = "^rec|^sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)


### STEP 2: EVALUATE THE BASE MODEL PERFORMANCE
sample_idx = 0
sample_row = preprocess_newsgroup_row(newsgroups_test.data[sample_idx])
sample_label = newsgroups_test.target_names[newsgroups_test.target[sample_idx]]

# print(sample_row)
# print('---')
# print('Label:', sample_label)

# Ask the model directly in a zero-shot prompt.

prompt = "From what newsgroup does the following message originate?"
baseline_response = client.models.generate_content(
    model="gemini-1.5-flash-001",
    contents=[prompt, sample_row])
# print(baseline_response.text)

# ask the model with instruction
system_instruct = """
You are a classification service. You will be passed input that represents
a newsgroup post and you must respond with the newsgroup from which the post
originates.
"""

@retry.Retry(predicate=is_retriable)
def predict_label(post: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash-001",
        config=types.GenerateContentConfig(
            system_instruction=system_instruct),
        contents=post)

    rc = response.candidates[0]

    # Any errors, filters, recitation, etc we can mark as a general error
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        # Clean up the response.
        return response.text.strip()

# # uncomment the following to try to test 1 example
# prediction = predict_label(sample_row)
# print(prediction)
# print()
# print("Correct!" if prediction == sample_label else "Incorrect.")

# Enable tqdm features on Pandas.
tqdmr.pandas()

# But suppress the experimental warning
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)


# Further sample the test data to be mindful of the free-tier quota.
df_baseline_eval = sample_data(df_test, 2, '.*')

# Make predictions using the sampled data.
df_baseline_eval['Prediction'] = df_baseline_eval['Text'].progress_apply(predict_label)

# And calculate the accuracy.
accuracy = (df_baseline_eval["Class Name"] == df_baseline_eval["Prediction"]).sum() / len(df_baseline_eval)
print(f"Accuracy: {accuracy:.2%}")
## uncomment to see the real label and predicted label for the dataset
# print (df_baseline_eval)


### STEP 3 : TUNING THE MODEL ###
# Convert the data frame into a dataset suitable for tuning.
input_data = {'examples': 
    df_train[['Text', 'Class Name']]
      .rename(columns={'Text': 'textInput', 'Class Name': 'output'})
      .to_dict(orient='records')
 }

# If you are re-running this lab, add your model_id here.
model_id = None

# Or try and find a recent tuning job.
if not model_id:
  queued_model = None
  # Newest models first.
  for m in reversed(client.tunings.list()):
    # Only look at newsgroup classification models.
    if m.name.startswith('tunedModels/newsgroup-classification-model'):
      # If there is a completed model, use the first (newest) one.
      if m.state.name == 'JOB_STATE_SUCCEEDED':
        model_id = m.name
        print('Found existing tuned model to reuse.')
        break

      elif m.state.name == 'JOB_STATE_RUNNING' and not queued_model:
        # If there's a model still queued, remember the most recent one.
        queued_model = m.name
  else:
    if queued_model:
      model_id = queued_model
      print('Found queued model, still waiting.')


# Upload the training data and queue the tuning job.
if not model_id:
    tuning_op = client.tunings.tune(
        base_model="models/gemini-1.5-flash-001-tuning",
        training_dataset=input_data,
        config=types.CreateTuningJobConfig(
            tuned_model_display_name="Newsgroup classification model",
            batch_size=16,
            epoch_count=2,
        ),
    )

    print(tuning_op.state)
    model_id = tuning_op.name

print(model_id)

### STEP 4 : TRACKING THE STATUS WHILE TUNING THE MODEL AT TEH BACKGROUND ###
MAX_WAIT = datetime.timedelta(minutes=10)

while not (tuned_model := client.tunings.get(name=model_id)).has_ended:

    print(tuned_model.state)
    time.sleep(60)

    # Don't wait too long. Use a public model if this is going to take a while.
    if datetime.datetime.now(datetime.timezone.utc) - tuned_model.create_time > MAX_WAIT:
        print("Taking a shortcut, using a previously prepared model.")
        model_id = "tunedModels/newsgroup-classification-model-ltenbi1b"
        tuned_model = client.tunings.get(name=model_id)
        break


print(f"Done! The model state is: {tuned_model.state.name}")

if not tuned_model.has_succeeded and tuned_model.error:
    print("Error:", tuned_model.error)

### STEP 5 : ONCE TUNING IS DONE, USE THE NEW MODEL ###
# # use one single example
# new_text = """
# First-timer looking to get out of here.

# Hi, I'm writing about my interest in travelling to the outer limits!

# What kind of craft can I buy? What is easiest to access from this 3rd rock?

# Let me know how to do that please.
# """

# response = client.models.generate_content(
#     model=model_id, contents=new_text)
# print("------ ONE EXAMPLE CHECK -----")
# print(response.text)
# print()


@retry.Retry(predicate=is_retriable)
def classify_text(text: str) -> str:
    """Classify the provided text into a known newsgroup."""
    response = client.models.generate_content(
        model=model_id, contents=text)
    rc = response.candidates[0]

    # Any errors, filters, recitation, etc we can mark as a general error
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        return rc.content.parts[0].text


# The sampling here is just to minimise your quota usage. If you can, you should
# evaluate the whole test set with `df_model_eval = df_test.copy()`.
df_model_eval = sample_data(df_test, 4, '.*')

df_model_eval["Prediction"] = df_model_eval["Text"].progress_apply(classify_text)

accuracy = (df_model_eval["Class Name"] == df_model_eval["Prediction"]).sum() / len(df_model_eval)
print(f"Accuracy: {accuracy:.2%}")

### STEP 6 : calculate the token used and compare ###
# Calculate the *input* cost of the baseline model with system instructions.
sysint_tokens = client.models.count_tokens(
    model='gemini-1.5-flash-001', contents=[system_instruct, sample_row]
).total_tokens
print(f'System instructed baseline model: {sysint_tokens} (input)')

# Calculate the input cost of the tuned model.
tuned_tokens = client.models.count_tokens(model=tuned_model.base_model, contents=sample_row).total_tokens
print(f'Tuned model: {tuned_tokens} (input)')

savings = (sysint_tokens - tuned_tokens) / tuned_tokens
print(f'Token savings: {savings:.2%}')  # Note that this is only n=1.
print ( "------------------------------------------")

baseline_token_output = baseline_response.usage_metadata.candidates_token_count
print('Baseline (verbose) output tokens:', baseline_token_output)

tuned_model_output = client.models.generate_content(
    model=model_id, contents=sample_row)
tuned_tokens_output = tuned_model_output.usage_metadata.candidates_token_count
print('Tuned output tokens:', tuned_tokens_output)


