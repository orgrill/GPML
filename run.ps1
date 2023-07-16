py -m ensurepip --default-pip
pip install -r requirements.txt

matlab -batch "run('src/main.m');exit;"