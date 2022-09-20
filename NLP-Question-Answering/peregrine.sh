module load TensorFlow/2.5.0-fosscuda-2020b
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load git/2.14.1-GCCcore-6.4.0

git clone https://github.com/SytseOegema/DeepLearning.git

python3 -m venv /data/$USER/.envs/bert_env

pip install --upgrade pip
pip install --upgrade wheel

pip install -r code/requirements.txt --user
