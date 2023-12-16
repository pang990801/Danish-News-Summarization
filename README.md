# Danish-News-Summarization
ITU Advanced Natural Language Processing and Deep Learning (2023) final project: Danish summarization model based on a dataset of automatically generated labels

## Workflow

### Step 1: Model Conversion
First, run the Ctranslate_converter.py script to obtain the opus-mt-en-da and opus-mt-da-en models converted using Ctranslate2.

'''
python Ctranslate_converter.py
'''

### Step 2: Translate Dataset
Use translate.py to translate the Danish news dataset into English.

'''
python translate.py
'''

### Step 3: Extract Summary
Next, run summary.py to extract summaries from the translated English news dataset.

'''
python summary.py
'''

### Step 4: Translate Back to Danish
Use translate_back.py to translate the extracted summaries back into Danish, resulting in a tagged Danish news dataset.

'''
python translate_back.py
'''

### Fine-Tuning the Model
Run finetune.py to fine-tune the mt5 model using the generated dataset.

'''
python finetune.py
'''

### Model Evaluation
Use use_model.py to observe the model's predictions, and use eval.py for model evaluation.

'''
python use_model.py
python eval.py
'''

## Contributing
Contributions to this project are welcome. Please discuss your proposed changes before submitting a pull request.
