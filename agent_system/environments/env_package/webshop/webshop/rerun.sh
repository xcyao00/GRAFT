#!/bin/bash

# Install Python Dependencies
pip install gym==0.24.0
pip install beautifulsoup4==4.11.1
pip install selenium==4.2.0
pip install cleantext==1.1.4
pip install rank_bm25==0.2.2
pip install pyserini==0.17.0
pip install thefuzz==0.19.0

# 替代 conda install mkl
pip install mkl
# 替代 conda install -c conda-forge faiss-cpu
pip install faiss-cpu

# 替代 conda install -c conda-forge openjdk=11
# yum install -y java-11-openjdk-devel 

# ubuntu安装
apt install -y openjdk-11-jdk

# Download spaCy large NLP model
pip install en_core_web_sm-3.7.1-py3-none-any.whl
pip install en_core_web_lg-3.7.1-py3-none-any.whl 


