from comet_ml import Experiment

API_KEY = "RoqFxUQ2dJHm8RjW1YatD0VQw" 
PROJECT_NAME = "MSI_Tank_DQN"
WORKSPACE = "jbuka"    

experiment = Experiment(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
        auto_output_logging="simple"
    )
experiment.end()