Servise is started with the command python3 -m uvicorn app:app --reload. 

The request can be sent in the format http://127.0.0.1:8000/post/recommendations/?id=2000&?limit=5
This request will recommend 5 post to the user with id 1000, which are most likely will be liked by this user.

Note: function load_tables(from_file=False) must be always run with from_file=False, as no files ar in the directory, due to large size.

Folder 'model learning' contains scripts for features engineering and
model learning, which are not necessary for service launch.