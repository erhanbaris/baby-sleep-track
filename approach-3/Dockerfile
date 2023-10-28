FROM python:3.11-slim
RUN apt-get update && apt -y install python3 python3-dev python3-pip libgl1-mesa-glx libglib2.0-0 g++ jq

EXPOSE 8501

WORKDIR /app
COPY requirements.txt /app
COPY run.sh /app
COPY face_landmarker_v2_with_blendshapes.task /app
COPY main.py /app
COPY utils.py /app
COPY streamlit_app.py /app

RUN pip install -r /app/requirements.txt

RUN chmod a+x /app/run.sh

CMD [ "/app/run.sh" ]
