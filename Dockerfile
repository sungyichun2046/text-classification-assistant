# we used the official python image
FROM python:3.9

# define the working directory of the docker image
WORKDIR /code

# copying file to the working directory
COPY ./requirements.txt /code/requirements.txt

# install library on the working directory
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# copying all of the necessary file and folder to the working directory
COPY ./app.py /code/app.py
COPY ./config.py /code/config.py

EXPOSE 8000

# command about how to run the app
CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.baseUrlPath=/Testweb/"]