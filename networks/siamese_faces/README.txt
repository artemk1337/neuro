1. name:
Name of the person identified in the image. Note that we only identify one person even though there may be more than one present in the image.

2. image_id: 
A number we associate with the image the face is found in. This number is unique but not necessarily in running sequence. Also, there may be multiple entries with the same image_id as there can be multiple faces of the same person in an image (e.g., where the image is an image composite). This happens even though the method described in our paper explicitly keeps at most one face per image. The additional faces are there because we manually add them back into the dataset in order not to waste them.

3. face_id: 
A number we associate with a particular face in our dataset. This number is unique for each face and is in running sequence.

4. url: 
The image file URL.

5. bbox: 
The coordinates of the bounding box for a face in the image. The format is x1,y1,x2,y2, where (x1,y1) is the coordinate of the top-left corner of the bounding box and (x2,y2) is that of the bottom-right corner, with (0,0) as the top-left corner of the image.  Assuming the image is represented as a Python Numpy array I, a face in I can be obtained as I[y1:y2, x1:x2].

6. sha256: 
The SHA-256 hash computed from the image.
After downloading an image from a given URL, the user can check if the downloaded image matches the one from our database by computing its SHA-256 hash and comparing it to the "sha256" value for the line.
On Linux systems, the SHA-256 hash can be obtained using the utility sha256sum in the shell.
