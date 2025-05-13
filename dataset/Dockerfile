FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
COPY emnist_download.py .env /workspace/
COPY env_path_fns.py /workspace/common/
CMD python emnist_download.py