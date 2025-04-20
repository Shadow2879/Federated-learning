FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
COPY emnist_download.py requirements.txt .env /workspace/
RUN pip install -r requirements.txt
CMD python emnist_download.py