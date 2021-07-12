
# Sentiment Classify
Classify reviewing sentences into positive or negative, assist restaurants, cafes to know customer feeling to enhance service quality.

## Installation

* Download and unzip to [trained_model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip).

* Download folder [last_weights](https://drive.google.com/drive/folders/16DqN5kKX1ji08HAqIxh4W_s7siNCFmkR?usp=sharing) that contains the weights was trained on [NTC_SV](train/) dataset.

* Install required modules with [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirement.txt
```
## Usage
Use [python](https://www.python.org/downloads/) for interpreting

```bash
python main.py 3000
```
## Customization
Use [Training.ipynb](train/Training.ipynb) and your dataset to train a new model being appreciated for your objective.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.