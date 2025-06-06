FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
COPY ./ /workspace/
RUN pip install -r requirements.txt
CMD python main.py && python run_agg.py