from ray import serve
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage
from starlette.requests import Request
import logging
import os
import base64
import json


ray_serve_logger = logging.getLogger("ray.serve")
MODEL_LOCAL_DIR = '/tmp/phi3'
DEVICE = 'cuda:0'  # 'auto'
MODEL_BUCKET_NAME = "apiiro-trained-models"
MODEL_BUCKET_DIRECTORY = "risky-feature-requests/phi-3/"


def load_model(model_path):
    ray_serve_logger.warning("Start Model loading ..")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    compute_dtype = torch.float32
    # device = torch.device(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=DEVICE,
        trust_remote_code=True
    )
    ray_serve_logger.warning(f"Model was loaded successfully.")
    return model, tokenizer


def get_next_word_probabilities(sentence, tokenizer, device, model, top_k=1):
    # Get the model predictions for the sentence.
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)  # .cuda()
    outputs = model(inputs)
    predictions = outputs[0]
    # Get the next token candidates.
    next_token_candidates_tensor = predictions[0, -1, :]
    # Get the top k next token candidates.
    topk_candidates_indexes = torch.topk(next_token_candidates_tensor, top_k).indices.tolist()
    # Get the token probabilities for all candidates.
    all_candidates_probabilities = torch.nn.functional.softmax(
        next_token_candidates_tensor, dim=-1)
    # Filter the token probabilities for the top k candidates.
    topk_candidates_probabilities = \
        all_candidates_probabilities[topk_candidates_indexes].tolist()
    # Decode the top k candidates back to words.
    topk_candidates_tokens = \
        [tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
    # Return the top k candidates and their probabilities.
    return list(zip(topk_candidates_tokens, topk_candidates_probabilities))


def get_first_line(file_path):
    """Reads and returns the first line of a file."""
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            return first_line
    except FileNotFoundError:
        return f"Error: The file {file_path} does not exist."
    except Exception as e:
        return f"Error: An unexpected error occurred. {str(e)}"


def download_directory(bucket_name, source_directory, destination_directory):
    """Downloads all files from a specified directory in a bucket to a local directory."""
    # Initialize a client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # List all blobs in the directory
    blobs = bucket.list_blobs(prefix=source_directory)
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    for blob in blobs:
        # Determine the local file path
        local_path = os.path.join(destination_directory, os.path.relpath(blob.name, source_directory))
        local_dir = os.path.dirname(local_path)
        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        # Download the file
        if source_directory != blob.name:
            ray_serve_logger.debug(f"Is-Risky-Feature - Start download model file {blob.name}.")
            blob.download_to_filename(local_path)


def get_risky_score(sentence, tokenizer, device, model):
    res = get_next_word_probabilities(sentence, tokenizer, device, model, top_k=1)
    chosen_res = res[0]
    return chosen_res[1] if chosen_res[0].lower() == 'pos' else 1 - chosen_res[1]


def is_input_valid(req):
    title = None
    description = None
    if 'title' in req and 'description' in req:
        title = req['title']
        description = req['description']
    if title is None or description is None \
        or "View in Apiiro" in description or "View full details in Apiiro" in description \
        or len(description.split(" ")) < 20:
            return None, None
    return title, description


@serve.deployment(ray_actor_options={"num_gpus": 1})
class RiskyFeatures:
    def __init__(self):
        encoded_key = os.getenv('GCP_CRED')
        if encoded_key is None:
            ray_serve_logger.error("Is-Risky-Feature - Fail to initialize model inference, on download model.  "
                                   "missing environment variable GCP_CRED")
        decoded_key = base64.b64decode(encoded_key).decode('utf-8')
        with open('/tmp/temp_credentials.json', 'w') as temp_file:
            temp_file.write(decoded_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/tmp/temp_credentials.json'
        ray_serve_logger.debug(f"Is-Risky-Feature - Start to download model from bucket = {MODEL_BUCKET_NAME}, "
                               f"path = {MODEL_BUCKET_DIRECTORY}  to local path = {MODEL_LOCAL_DIR}")
        download_directory(MODEL_BUCKET_NAME, MODEL_BUCKET_DIRECTORY, MODEL_LOCAL_DIR)
        ray_serve_logger.debug(f"Is-Risky-Feature - Model download finished.")

        ray_serve_logger.debug(f"Is-Risky-Feature - Start to load model from {MODEL_LOCAL_DIR}.")
        self.model, self.tokenizer = load_model(MODEL_LOCAL_DIR)
        ray_serve_logger.debug(f"Is-Risky-Feature - Model loading complete.")

    async def __call__(self, request: Request) -> str:
        req = await request.json()
        confidence = 0
        sentence = None
        title, description = is_input_valid(req)
        if title is not None and description is not None:
            try:
                sentence = title + " " + description
                ray_serve_logger.debug(f"Is-Risky-Feature input is {sentence}")
                confidence = get_risky_score(sentence, self.tokenizer, DEVICE, self.model)  # cuda:0
            except Exception as e:
                ray_serve_logger.error(f"Error in Is-Risky-Feature. for input {sentence}.")
                ray_serve_logger.error(f"Error in Is-Risky-Feature. {e}.")
        else:
            ray_serve_logger.error(f"Missing input fields in the json request = {req}")
        ray_serve_logger.debug(f"Is-Risky-Feature confidence result is {confidence}")
        result = json.dumps({"issueRiskPredictionConfidence": confidence})
        return result


deployment_graph = RiskyFeatures.bind()
