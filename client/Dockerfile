FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
COPY ./ /workspace/
RUN pip install -r requirements.txt
CMD python main.py