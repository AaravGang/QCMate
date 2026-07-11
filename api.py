# 1. Import the library
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
load_dotenv()

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW")
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="aarav-gang",
    workflow_id="macro-level-drawing-annotations-vdrawing-annotations-logic",
    images={
        "image": "datasets/v2_1024x1024/FCF_DATASET_1024/test/images/drawing23Images_2046114220_drw-1_png.rf.65e3b496d9c931db64f2c06849269887.jpg"  # Path to your image file
    },
    use_cache=True  # Speeds up repeated requests
)

# 4. Get your results
print(result)
