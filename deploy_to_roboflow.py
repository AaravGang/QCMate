from roboflow import Roboflow
from dotenv import load_dotenv
import os 
load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW"))
workspace = rf.workspace("aarav-gang")

workspace.deploy_model(
    model_type="yolov11s",  # Type of the model
    model_path="models/v2",  # Path to model directory
    project_ids=["macro-level-drawing-annotations"],  # List of project IDs
    # Name for the model (must have at least 1 letter, and accept numbers and dashes)
    model_name="drawing-annotations",
    filename="weights/best.pt"  # Path to weights file (default)
)
