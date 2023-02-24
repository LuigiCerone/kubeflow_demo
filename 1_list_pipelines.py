import kfp
import json

host = 'localhost:8080'
pipeline_name = '[Demo] XGBoost - Iterative model training'

client = kfp.Client()

filter = json.dumps({'predicates': [{'key': 'name', 'op': 1, 'string_value': '{}'.format(pipeline_name)}]})

pipelines = client.pipelines.list_pipelines(filter=filter)

print(pipelines)
