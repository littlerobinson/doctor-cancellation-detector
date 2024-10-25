# doctor-cancellation-predict

Machine learning model with MLflow environment to detect cancellation with doctor appointment.

## Install

Do this command in the docker folder.

```bash
heroku container:login
heroku create --region eu app_name
heroku stack:set container -a  app_name
heroku container:push web -a app_name
heroku container:release web -a app_name
```

## Launch training

### With a script

Create for example a run.sh file with this code :

```bash
docker run -it -p 8080:8080\
 -v "$(pwd):/mlflow"\
 -e PORT=8080\
 -e AWS_ACCESS_KEY_ID="xxx"\
 -e AWS_SECRET_ACCESS_KEY="xxx"\
 -e AWS_ARTIFACT_S3_URI="s3://xxx-bucket/xxx-artifacts/"\
 -e DATABASE_URL="postgres://xxxx"\
 mlflow-container-name python train.py
```

### With a MLproject file

Create a secrets.sh scripts with all env data to export.

```bash
export MLFLOW_TRACKING_URI="https://mlflow-xxx.herokuapp.com"
export ...
```

Add to OS env variables.

```bash
source secrets.sh
```

launch :

```bash
mlflow run .
```
