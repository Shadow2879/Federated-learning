FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
COPY ./ /workspace/
RUN pip install -r requirements.txt
CMD python main.py