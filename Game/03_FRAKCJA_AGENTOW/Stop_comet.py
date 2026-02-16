import comet_ml

API_KEY = "L2PzW7c3YM3WqM5hNfCsloeLZ" 
PROJECT_NAME = "msi-projekt"
WORKSPACE = "kluski777"

experiment = comet_ml.start(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE
    )
experiment.end()