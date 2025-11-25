#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
clear
python3 get_data.py
python3 tfidf_example.py
python3 faiss_example.py