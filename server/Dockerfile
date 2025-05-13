FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
COPY requirements.txt main.py /workspace/
RUN pip install -r requirements.txt
CMD python main.py