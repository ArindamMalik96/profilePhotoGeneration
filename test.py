import time

start = time.time()
import os

url="https://www.jeevansathi.com/uploads/NonScreenedImages/mainPic/82/187/82023131iibe5c8acb642ca4ec602cb5e80ac05fe5ii100e4f636ba8f03245d9832e5ac19f1e.jpeg"

#filename_w_ext = url[url.rfind("/")+1:]
#ind=filename_w_ext.index('.')
#filename=filename_w_ext[:ind]
#file_extension=filename_w_ext[ind:]

filename_w_ext = os.path.basename(url)
filename, file_extension = os.path.splitext(filename_w_ext)


print("\nfilename:",filename_w_ext,"##",filename,"##",file_extension,"\n")

end = time.time()
print("\n",(end - start)*10)
 
