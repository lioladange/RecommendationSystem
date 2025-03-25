This project is a RestAPI service, which recommends posts to the users of a social
network, based on predictions by catboost model. 
Model is trained on user data, post data and history of their interactions.

Service has one available request, which requires user id, as mandatory parameter.
Amount of recommended posts and time, when request was sent, can be also sent
as a parameter (limit and time) by default limit=5 and time=''.

### Project launch:
1. To install necessary environment, run:

`pip install -r requirements.txt`

2. To run the service locally, use:

`python3 -m uvicorn app:app --reload.`

3. Send request, using Postman browser, python "requests" library, usual browser or other
available tool for requests.
The request can be sent in the format http://127.0.0.1:8000/post/recommendations/?id=2000&?limit=5
This request will send request to the localhost, which will recommend 5 post to the user 
with id 1000, which are most likely will be liked by this user.



__Note:__ function load_tables(from_file=False) must be always run with from_file=False, 
as no files at in the directory, due to large size. 

Folder 'model learning' contains scripts for features engineering and
model learning, which are not necessary for service launch.
