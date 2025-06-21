FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
COPY ./.env /workspace/
RUN pip install -r requirements.txt
COPY ./ /workspace/
CMD python main.py