# Effortless profile photo selection

## Problem 🌍 
With so many online face detection software running, it is hard to find just the right one that suit your need. This model will provide you cropped image from a url, that will focus on ***the most prominent person***, ***reject photo that has multiple faces of equal size*** and ***reject photo that has only small faces***. This code runs on docker so by following just couple of lines you will have complete running model that you attach to your application/platform.

## Installation instructions ‍ 
The project run inside docker container so you do not need to worry about any installation. Just follow simple instructions.

Download the project using
```bash
git clone https://github.com/ArindamMalik96/profilePhotoGeneration.git
```
### Installation and playing with docker images

Install docker if already not installed through below link
Linux [🔗](https://docs.docker.com/engine/install/ubuntu/)
Windows [🔗](https://docs.docker.com/docker-for-windows/install/)
MacOS [🔗](https://docs.docker.com/docker-for-mac/install/)


### Go inside the root folder
```
cd profilePhotoGeneration
```

### Create docker image from tar inside 
#### Linux
```
docker load -i profilePhotoGenerationDocker.tar
```

Check if docker image has been successfully created:
```
docker images
``` 
Let's say your docker image name is gender_detection3

Use following command to run docker instance 
```
docker run --name faceDetectionapp01 -p 8013:80 -v profilePhotoGeneration/:/app -e FLASK_DEBUG=0 -e NGINX_WORKER_PROCESSES=auto -e PYTHONPATH=$PYTHONPATH:/app/facenet/src -dit gender_detection3
```
### Want to try the API you created? Try the API you created 
#### POST url:
```
localhost:8013/api/detectCaffeFace
```
#### Message Body:
```
{"imgUrl" : "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/%D0%9C%D0%B0%D0%BB%D1%8C%D1%87%D0%B8%D0%BA_%28%D0%BF%D1%80%D0%BE%D1%84%D0%B5%D1%81%D1%81%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5_%D1%84%D0%BE%D1%82%D0%BE%29.jpg/220px-%D0%9C%D0%B0%D0%BB%D1%8C%D1%87%D0%B8%D0%BA_%28%D0%BF%D1%80%D0%BE%D1%84%D0%B5%D1%81%D1%81%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5_%D1%84%D0%BE%D1%82%D0%BE%29.jpg"
}
```

#### You can also use the following curl for ease:
```
curl --location --request POST '127.0.0.1:8013/api/perfectProfilePhoto' \
--header 'Content-Type: application/json' \
--data-raw '{"imgUrl" : "https://img.srgcdn.com/e/w:1000/M3JFd0tENzBoaFJDQ0UzT05paGUuanBn.jpg"
}'
```

This will return cropped image


## Stuck somewhere?
Please feel free to reach out to me here:
[Arindam Malik](mailto:arindammalik96@gmail.com)
