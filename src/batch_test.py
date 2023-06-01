import os
import subprocess
import config

# Automate running saliency.py on multiple molecules

# List of molecules to run saliency on
CAS_list=["5306-98-9","3964-56-5","50262-77-6","112-53-8","2432-14-6","2896-60-8","5349-63-3","371-41-5","1460-57-7","2434-49-3","1195-09-1",    "141-97-9","20383-28-2","4466-24-4","312-94-7","4119-41-9","3624-90-6","24083-13-4","15869-85-9","7145-23-5","25784-91-2","20389-01-9"]
#CAS_list=["141-97-9","20383-28-2","4466-24-4","312-94-7","4119-41-9","3624-90-6","24083-13-4","15869-85-9","7145-23-5","25784-91-2","20389-01-9"]

for i in CAS_list:
    print(i)
    os.system(f"sudo python {config.ROOT_PATH}/src/saliency.py {i}")


