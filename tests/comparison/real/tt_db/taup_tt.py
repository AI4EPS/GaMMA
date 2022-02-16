import math
import obspy.taup
import numpy as ny
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
build_taup_model("itvel.nd")
model = TauPyModel(model="itvel")

with open("ttdb.txt", "w") as f:
    #f.write("dist dep tp ts tp_slowness ts_slowness tp_hslowness ts_hslowness p_elvecorr s_elvecorr\n")
    for dep in range(0,21,1): # search 0 - 20 km in depth
        for dist in range(1,101,1): # search 0.01 - 2.0 degree in horizontal 
            dist = dist*0.01
            print(dep,dist)
            arrivals = model.get_travel_times(source_depth_in_km=dep, distance_in_degree=dist, phase_list=["P","p","S","s"])
            #print(arrivals)model
            i = 0
            pi = 0
            si = 0
            while(i<len(arrivals)):
                arr = arrivals[i]
                i = i + 1
                if((arr.name == 'P' or arr.name == 'p') and pi == 0):
                    pname = arr.name
                    p_time = arr.time
                    p_ray_param = arr.ray_param*2*ny.pi/360
                    p_hslowness = -1*(p_ray_param/111.19)/math.tan(arr.takeoff_angle*math.pi/180)
                    pi = 1

                if((arr.name == 'S' or arr.name == 's') and si == 0):
                    sname = arr.name
                    s_time = arr.time
                    s_ray_param = arr.ray_param*2*ny.pi/360
                    s_hslowness = -1*(s_ray_param/111.19)/math.tan(arr.takeoff_angle*math.pi/180)
                    si = 1
                if(pi == 1 and si == 1):
                    break
                    
            f.write("{} {} {} {} {} {} {} {} {} {}\n".format(dist, dep, p_time,s_time, p_ray_param, s_ray_param, p_hslowness, s_hslowness, pname, sname))
