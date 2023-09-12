# BTP_ULL

Steps to run the code:
1. pip install all the libraries in the requirements.txt file (there may be other dependencies which are taken care of in colab but might have to be installed for other servers)
2. get the data by running the bash file using the command '''sh get_data.sh'''
3. run the trainer script using '''CUDA_LAUNCH_BLOCKING=1 python trainer.py'''
