from os import path, listdir, remove
from os.path import join
from keras.models import Sequential, load_model, model_from_json
import json
import pprint
import sys

folder_path = sys.argv[1]

model_paths = [join(folder_path, f) for f in listdir(folder_path) if f.endswith("h5")]
print(model_paths)

for path in model_paths:
    model = load_model(path)
    json_model = model.to_json()
    jjson_model = json.loads(json_model)
    jjson_model['config'][4]['config']['recurrent_initializer'] = {'class_name': 'Zeros', 'config': {}}
    model.save_weights('weights_tmp.h5')
    model = model_from_json(json.dumps(jjson_model))
    model.load_weights('weights_tmp.h5')
    remove('weights_tmp.h5')
    model.save(path)
