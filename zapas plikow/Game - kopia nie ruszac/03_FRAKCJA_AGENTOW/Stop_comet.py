from comet_ml import Experiment

API_KEY = "L2PzW7c3YM3WqM5hNfCsloeLZ" 
PROJECT_NAME = "msi-projekt"
WORKSPACE = "kluski777"

experiment = Experiment(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
        auto_output_logging="simple"
    )
experiment.end()