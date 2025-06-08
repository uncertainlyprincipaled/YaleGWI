# GWI 2025 - Velocity Patterns & Training Dataframe

# # GWI 2025 - Velocity Patterns and a Training Dataframe
# The Yale/UNC-CH - Geophysical Waveform Inversion competition is a bit intimidating: abstractly, the task is to learn how to transform 5 given 1000x70 input images into one corresponding 70x35 output image. There are large, pretrained ML models already created which seem to do very well on the clean, synthetic benchmark data. So, here I'm taking a simple tabular approach to make crude predictions.

# To get an idea of the data, images of the velocity maps and seismic data are made with simple plots based on the tutorial notebook. One novelty is to scale the seismic data by f(t) = 1 + (t/a)^b (with, e.g., a=200 ms, b=1.5) to help equalize the amplitudes vs time; this is similar to applying AGC when visualizing.

# A Training Dataframe is created and used as a place to add properties of the velocity maps (the targets) and the seismic data (the features). So far the only seismic features extracted are the surface velocities in the left (0-34) and right (34-69) halves combined to give Average and R-L Difference velocities for each sample. (The velocities are accurately measured using times based on polyfits to points around peaks. Quickly-reflected waves can affect the measurements.)

# Scatter plots of the Average vs R-L Difference seismic velocities show an interesting pattern, similar for both the Train and Test data. The plots divide into samples with the surface velocity R-L difference relatively near 0 (blue points) and ones with asymmetric surface velocities (red points). The differences are probably due to the different vmap types: some are symmetric, some are not.

# The targets to predict for each sample are the median training vmap values in coarse horizontal row ranges: 0-9 L&R, 10-29, 30-49, and 50-69. Correctly predicting all 5 medians for each sample would give MAE ~ 250. Simple polynomial models are fit to the vmap median values (y) versus the seismic average surface velocity (x) separately for the blue and red points. To approximate an MAE metric, the points are converted to median points in quantile ranges (by xy_medians()). These models are then used to make test predictions based on the measured test surface velocities.

# Notes:
#   The 5 sources are x-located closest to: 0, 17, 34, 52, 69

# Looked through the public notebooks, noted these:
#   _This notebook has nice image-making code for velocity and data:
#     https://www.kaggle.com/code/hanchenwang114/waveform-inversion-kaggle-competition-tutorial 
#   _"Muting" surface pulses is done in: https://www.kaggle.com/code/ozhiro/topmute-analysis 
#     Looks like "AGC" ~ adjusts gain within a (range of) row(s). Instead, I'll scale the data by the time.
#   _This notebook mentions multiply-reflected waves, I wonder if they are seen/important :
#     https://www.kaggle.com/code/nikita7364777/u-net-lb-413 

# Some things to do next:
# . . .

# Usual Things to Use 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Useful Functions
# To show and process the Velocity Maps

# Plot the velocity map
def plot_velocity(velocity, sample):
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    img=ax.imshow(velocity[sample,0,:,:],cmap='jet')
    ax.set_xticks(range(0, 70, 10))
    ax.set_xticklabels(range(0, 700, 100))
    ax.set_yticks(range(0, 70, 10))
    ax.set_yticklabels(range(0, 700, 100))
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_xlabel('Offset (m)', fontsize=12)
    clb=plt.colorbar(img, ax=ax)
    clb.ax.set_title('km/s',fontsize=8)
    plt.show()
    # And a simple ave velocity vs depth plot
    plt.figure(figsize=(8, 2.5))
    plt.plot(np.arange(5,700,10), np.mean(velocity[sample,0,:,:],axis=1))
    plt.xlabel("Depth (m)")
    plt.ylabel("Ave Velocity (m/s)")
    plt.show()

# Get information from the velocity map
def info_velocity(velocity, sample, for_show=True):
    # When for_show=True display results and plots.
    # When for_show=False work silently and return measured values.
    # Indices are: sample, 0, depth, xloc
    ave_vel = np.mean(velocity[sample,0,:,:])
    std_vel = np.std(velocity[sample,0,:,:])
    min_vel = np.min(velocity[sample,0,:,:])
    max_vel = np.max(velocity[sample,0,:,:])
    medi_vel = np.median(velocity[sample,0,:,:])
    MAE_1medi = np.mean(np.abs(velocity[sample,0,:,:] - medi_vel))
    # Number of unique velocities
    num_vels = len(np.unique(velocity[isample,0,:,:]))
    # Average velocities in first row halves ~ surface velocity
    y0_velL = np.mean(velocity[sample,0, 0 , 0:35  ])
    y0_velR = np.mean(velocity[sample,0, 0 , 35:  ])
    # Median velocities in rows 0-9, 10-29, 30-49, 50-69
    # and keep track of MAE wrt to these
    MAE_5medi = 0.0
    y09L_medi = np.median(velocity[sample,0, 0:10 , 0:34+1  ])
    MAE_5medi += 5.0*np.mean(np.abs(velocity[sample,0, 0:10 , 0:34+1  ] - y09L_medi))
    y09R_medi = np.median(velocity[sample,0, 0:10 , 35:  ])
    MAE_5medi += 5.0*np.mean(np.abs(velocity[sample,0, 0:10 , 35:  ] - y09R_medi))
    y1029_medi = np.median(velocity[sample,0, 10:29+1 , :  ])
    MAE_5medi += 20.0*np.mean(np.abs(velocity[sample,0, 10:29+1 , :  ] - y1029_medi))
    y3049_medi = np.median(velocity[sample,0, 30:49+1 , :  ])
    MAE_5medi += 20.0*np.mean(np.abs(velocity[sample,0, 30:49+1 , :  ] - y3049_medi))
    y5069_medi = np.median(velocity[sample,0, 50: , :  ])
    MAE_5medi += 20.0*np.mean(np.abs(velocity[sample,0, 50: , :  ] - y5069_medi))
    MAE_5medi = MAE_5medi / 70.0
    # Means
    y09L_mean = np.mean(velocity[sample,0, 0:10 , 0:34+1  ])
    y09R_mean = np.mean(velocity[sample,0, 0:10 , 35:  ])
    y1029_mean = np.mean(velocity[sample,0, 10:29+1 , :  ])
    y3049_mean = np.mean(velocity[sample,0, 30:49+1 , :  ])
    y5069_mean = np.mean(velocity[sample,0, 50: , :  ])
    if for_show:
        print("Number of distinct velocities: {}".format(num_vels))
        print("Average velocity: {:.2f} m/s  SD: {:.2f}".format(ave_vel, std_vel))
        print("Median velocity: {:.2f} m/s".format(medi_vel),
             "   Min, Max: {:.2f}, {:.2f}".format(min_vel, max_vel))
        print("MAE from median: {:.2f}  ".format(MAE_1medi))
        print("Ave y=0 velocities L,R: {:.2f}, {:.2f}".format(y0_velL, y0_velR))
        print("Median velocities in rows:  {:.2f}(0-9:L), {:.2f}(0-9:R),".format(
                y09L_medi, y09R_medi),
                "{:.2f}(10-29), {:.2f}(30-49), {:.2f}(50-69)".format(
                y1029_medi, y3049_medi, y5069_medi))
        print("MAE from 5 medians: {:.2f}".format(MAE_5medi))
        print("  Mean velocities in rows:  {:.2f}(0-9:L), {:.2f}(0-9:R),".format(
                y09L_mean, y09R_mean),
                "{:.2f}(10-29), {:.2f}(30-49), {:.2f}(50-69)".format(
                y1029_mean, y3049_mean, y5069_mean))
        
    else:
        return (num_vels, y0_velL, y0_velR, y09L_medi, y09R_medi,
                    y1029_medi, y3049_medi, y5069_medi, MAE_1medi, MAE_5medi)

# To show and process the Seismic Data

# Make a gray-scale image of the seismic data
def plot_data(data, sample=-1):
    fig,ax=plt.subplots(1,5,figsize=(20,7))
    # Is it a Train (multiple) or Test (single) data?
    if len(data.shape) == 3: 
        thisdata = data[:,:,:]
    else:
        thisdata = data[sample,:,:,:]
    # Scale the color range. Use symmetric values to have 0 in the middle.
    # Use the values in the source columns to avoid source pulses.
    maxabs = []
    for srclocid, xloc in enumerate([0,17,34,52,69]):
        maxabs.append(np.max(np.abs(thisdata[srclocid,180:,xloc])))
    vrange = np.max(maxabs) * 0.5  # use less than max, some saturation is OK
    for iax in range(5):
        ax[iax].imshow( thisdata[iax,:,:], extent=[0,70,1000,0],
                       aspect='auto', cmap='gray', vmin=-vrange, vmax=vrange)
    for axis in ax:
       axis.set_xticks(range(0, 70, 10))
       axis.set_xticklabels(range(0, 700, 100))
       axis.set_yticks(range(0, 2000, 1000))
       axis.set_yticklabels(range(0, 2,1))
       axis.set_ylabel('Time (s)', fontsize=12)
       axis.set_xlabel('Offset (m)', fontsize=12)
    plt.show()


# Get an accurate time of the max (usually first) peak from given source in given xloc.
def time_max_peak(isrc, xloc, thisdata):
    ipeak = np.argmax(thisdata[isrc,:,xloc])
    # fit 7 points with degree=2
    peakvals = thisdata[isrc, ipeak-3:ipeak+3+1, xloc]
    timevals = np.linspace(ipeak-3,ipeak+3, num=7, endpoint=True)
    if len(peakvals) == len(timevals):
        fitcoefs = np.poly1d(np.polyfit(timevals, peakvals, 2)).coef
        # max is at -b/(2a)   :)
        return -0.5*fitcoefs[1]/fitcoefs[0]
    else:
        print("mis-matched lengths:\n",timevals, "\n", peakvals)
        return ipeak


# Get information from the seismic data
# When for_show=True display results and plots.
# When for_show=False work silently and return measured values.
def info_data(data, sample=-1, for_show=True):
    # Train (multiple) or Test (single) data?
    if len(data.shape) == 3:
        thisdata = data[:,:,:]
    else:
        thisdata = data[sample,:,:,:]
    # Calculate the surface velocity, don't use source columns
    # Ignore peaks that are too distant to be the surface peak at 200 m away from source:
    # Times less than: 225 ms. From: 100 ms (source peak) + 100 m / 800 m/s * 1000 s/ms.
    # Still have issues with quick reflections messing up the velocities -
    # use short baselines and average 4 on each side.
    partdata = thisdata[ : , 0:225 ,: ]
    vsurfaceL = 4*5*10*1000/( time_max_peak(0, 6, partdata) - time_max_peak(0, 1, partdata) +
                            time_max_peak(1, 11, partdata) - time_max_peak(1, 16, partdata) +
                            time_max_peak(1, 23, partdata) - time_max_peak(1, 18, partdata) +
                            time_max_peak(2, 28, partdata) - time_max_peak(2, 33, partdata) )
    vsurfaceR = 4*5*10*1000/( time_max_peak(2, 40, partdata) - time_max_peak(2, 35, partdata) +
                            time_max_peak(3, 46, partdata) - time_max_peak(3, 51, partdata) +
                            time_max_peak(3, 58, partdata) - time_max_peak(3, 53, partdata) +
                            time_max_peak(4, 63, partdata) - time_max_peak(4, 68, partdata) )
    # Clip the average velocity to 4100 - don't trust higher values are real.
    vsurface = np.clip((vsurfaceL + vsurfaceR)/2, 1400.0, 4100.0)
    if for_show:
        # Make a plot of surface wave distance vs time
        # Use the middle source location to avoid/reduce reflected wave interference
        idists = np.arange(0,70)
        dists = []
        times = []
        # Time is relative to src 2 peak time.
        timeref = np.argmax(thisdata[2,:,34])
        for idist in idists:
            dists.append(10*idist) # 0 to 690
            # signed time for before/after xloc=34
            times.append(np.sign(idist-34)*(np.argmax(thisdata[2,:,idist])-timeref)/1000.0)
        times = -1.0*(times - times[0])  # adjust orientation of time axis
        plt.figure(figsize=(6,3))
        plt.plot(dists, times, '.b', alpha=0.7)
        plt.plot([dists[0],dists[-1]],[times[0],times[-1]],c='orange',alpha=0.6)
        plt.ylabel("$-$ Time (s)")
        plt.xlabel("Surface Distance (m)")
        plt.title("Time vs Distance from Source 2")
        plt.show()
        ##v_orange = -1.0*(dists[-1] - dists[0])/(times[-1] - times[0])
        print("Surface velocities : {:.2f}-Left, {:.2f}-Average, {:.2f}-Right".format(
                        vsurfaceL, vsurface, vsurfaceR))
    else:
        return vsurfaceL, vsurface, vsurfaceR
    
# Make a plot of the waveforms at each of the source locations when that source is active.
def sources_data(data, sample=-1, for_show=True):
    # The 5 sources are located closest to: 0, 17, 34, 52, 69
    # The peak amplitude ~ 40 for each.
    # Train (multiple) or Test (single) data?
    if len(data.shape) == 3:
        thisdata = data[:,:,:]
    else:
        thisdata = data[sample,:,:,:]
    # Get the max, min amplitudes for t > 180 for each source-location
    maxamps = []
    minamps = []
    for srclocid, xloc in enumerate([0,17,34,52,69]):
        ##print("Source peak at xloc={} is: {:.2f}".format(
        ##        xloc, np.max(thisdata[srclocid,:,xloc]) ))
        # Max and min after the source peak
        maxamps.append(np.max(thisdata[srclocid,180:,xloc]))
        minamps.append(np.min(thisdata[srclocid,180:,xloc]))
    max_amp = np.max(maxamps)
    min_amp = np.min(minamps)
    delta_amp = 0.05*(max_amp - min_amp)
    plt.figure(figsize=(8,5))
    for srclocid, xloc in enumerate([0,17,34,52,69]):
        timeseries = thisdata[srclocid,:,xloc]  # srclocid, time, xloc
        offset = delta_amp*(xloc - 34)/35.0
        plt.plot(np.array(range(1000)) + 0*offset, timeseries + offset, alpha=0.7) 
    plt.plot([0,1000],[0.0,0.0],c='gray',alpha=0.5)
    plt.ylim(1.10*min_amp - delta_amp, 1.10*max_amp + delta_amp)
    plt.xlabel('Time (ms)')
    plt.ylabel("Amplitude     Traces are offset.")
    plt.title("Waveforms at the 5 source locations")
    plt.show()

# Routine to read in the training data files given the training dataframe index value

# There are two directory formats for getting the data-velocity pairs depending on the type:

# FlatVel_[A,B], CurveVel_[A,B], Style_[A|B]
# Each of these 6 dirs contain: /data/data[1,2].npy and /model/model[1,2].npy
# Total # of velocity-meaurement pairs: 12 x 500
##velocity = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy')
##data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data2.npy')
##isample = 13 

# [FlatFault,CurveFault]_A has files: seis[2,4]_1_0.npy, vel[2,4]_1_0.npy
# [FlatFault,CurveFault]_B has files: seis[6,8]_1_0.npy, vel[6,8]_1_0.npy
# Total # of velocity-meaurement pairs: 8 x 500
##velocity = np.load('/kaggle/input/waveform-inversion/train_samples/CurveFault_A/vel4_1_0.npy')
##data = np.load('/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis4_1_0.npy')
##isample = 23

# Keep track of the last train file read in to avoid re-reading when just isample changes
last_data_file = "None"

def get_train_sample(dfind, ftscale=True):
    # Assumes traindf is defined.  And uses global values:
    global velocity, data, last_data_file
    train_dir = "/kaggle/input/waveform-inversion/train_samples/"
    veltype, ifile, isample = traindf.loc[dfind, ["veltype","ifile","isample"]]
    if ("Vel" in veltype) or ("Style" in veltype):
        data_file = train_dir+veltype+"/data/data"+str(ifile)+".npy"
        model_file = train_dir+veltype+"/model/model"+str(ifile)+".npy"
        ##print("got Vel or Style:\n   ", data_file, "\n   ", model_file)
    else:  # it is a Fault type
        fault_num = 2*ifile + 4*("_B" in veltype)
        data_file = train_dir+veltype+"/seis"+str(fault_num)+"_1_0.npy"
        model_file = train_dir+veltype+"/vel"+str(fault_num)+"_1_0.npy"
        ##print("got Fault:\n   ", data_file, "\n   ", model_file)
    # Read them in if not already available
    if data_file != last_data_file:
            data = np.load(data_file)
            # Scale the seismic data as a function of time:
            if ftscale:
                for itime in range(1000):
                    data[ : , : , itime, : ] = (1.0+(itime/200)**1.5)*data[ : , : , itime, : ]
            velocity = np.load(model_file)
            last_data_file = data_file
    return velocity, data, isample

# Convert many x,y points into a quartile-based set of x_median,y_median points.
# Fitting these median points is similar to fitting x,y with a MAE metric.
def xy_medians(xin, yin, nqs):
    # Outputs x,y medians from about nqs quartiles
    sortinds = np.argsort(xin)
    xsort = xin[sortinds]
    ysort = yin[sortinds]
    lenxs = len(xsort)
    nsample = int(lenxs/nqs)
    # Have a first and last range of ~ nsample/3 points
    nfirstlast = int(nsample/4)
    indups = list(range(nfirstlast, lenxs - nfirstlast, nsample))
    indups.insert(0,0) # start with 0
    indups.append(lenxs - nfirstlast)
    indups.append(lenxs)
    ##print(indups)
    xmeds = []; ymeds = []
    for iup in range(0, len(indups)-1):
        indlow = indups[iup]
        indhi = indups[iup+1]
        xmed = np.median(xsort[indlow:indhi])
        ymed = np.median(ysort[indlow:indhi])
        xmeds.append(xmed)
        ymeds.append(ymed)
    # Add a value at xmax: average of last value and linear trend by quantile
    xmeds.append(xsort[-1])
    ymeds.append(ymeds[-1] + 0.5*(ymeds[-1] - ymeds[-2]))
    return xmeds, ymeds

# Make an Initial Dataframe
# There are 10,000 training samples on kaggle, organized as: 10 x 2 x 500 data-vel pairs
##!ls /kaggle/input/waveform-inversion/train_samples/*

# Make a dataframe with 10,000 rows labeled by:
#   type - 5 x 2 string values
#   ifile - two numeric values: 0,1 or 1,2 or 2,4 or 6,8 depending on type
#   isample - 0 to 499
veltypes = ["FlatVel","FlatFault", "CurveVel", "CurveFault", "Style"]
veltype = []; ifile = []; isample = []
for this_type in veltypes:
    for this_AB in ["_A","_B"]:
        for this_ifile in [1,2]:
            for this_isample in range(500):  # **************************************
                veltype.append(this_type+this_AB); ifile.append(this_ifile); isample.append(this_isample)
# Make a dataframe from these
traindf = pd.DataFrame({"veltype":veltype, "ifile":ifile, "isample":isample})

# Look at a Training Velocity-Data Pair
# Select a dataframe index to look at
dfind = int(0.91*len(traindf))


print(list(traindf.loc[dfind,["veltype","ifile","isample"]]))
velocity, data, isample = get_train_sample(dfind)

print('Velocity map size:', velocity.shape)
print('Seismic data size:', data.shape)
# Velocity map size: (500, 1, 70, 70)   sample, 0, yloc(10m), xloc(10m)
# Seismic data size: (500, 5, 1000, 70) sample, srclocid, time(ms), xloc(10m)

# Look at the velocity map for the training sample
# isample defined above
plot_velocity(velocity, isample)
info_velocity(velocity, isample)

# Look at the seismic data for the sample
# isample = same as for the velocity map above
plot_data(data, isample)
info_data(data, isample)
sources_data(data, isample)

# Look at a Test Data Sample
# The test data consists of 65818 data sets to be predicted
## !ls /kaggle/input/waveform-inversion/test | wc
# 65818   65818  987270

##!ls -s /kaggle/input/waveform-inversion/test/c*.npy | head -5
# total 90039024
# 1368 000039dca2.npy
# 1368 0000fd8ec8.npy
# 1368 0001026c8a.npy
# 1368 00015b24d5.npy
#      a00269f1eb.npy
#      c001726adb.npy
#      c0021521e5.npy

# Look at one of them (they seem to be shuffled)
##testdata = np.load('/kaggle/input/waveform-inversion/test/000039dca2.npy')  # messy
##testdata = np.load('/kaggle/input/waveform-inversion/test/0001026c8a.npy')  # very simple
testdata = np.load('/kaggle/input/waveform-inversion/test/00015b24d5.npy')  # weird straight lines
##testdata = np.load('/kaggle/input/waveform-inversion/test/800222ab0d.npy')  # messy
##testdata = np.load('/kaggle/input/waveform-inversion/test/a00269f1eb.npy')  # messy
##testdata = np.load('/kaggle/input/waveform-inversion/test/c0021521e5.npy')  # simple-ish

# Scale the seismic data by ~ (1+(t/a)^b) to help equalize the amplitudes vs time.
# (This is similar to applying AGC for visualization, but is included in the analysis too.)
for itime in range(1000):
    testdata[ : , itime, : ] = (1.0+(itime/200)**1.5)*testdata[ : , itime, : ]

print('Test data size:', testdata.shape)

plot_data(testdata)
info_data(testdata)
sources_data(testdata)

# Fill the TRaining Dataframe
# For each sample add:

# The y_ targets: y0_velL, y0_velR, y09L_medi, y09R_medi, y1039_medi, y4069_medi
nunique = []; y0_aves = []; y0_diffs = []
y09L_medis = []; y09R_medis = []; y1029_medis = []; y3049_medis = []; y5069_medis = []

# These properties of the target velocity map
MAE_1medis = []; MAE_5medis = []

# The x_ features: surface velocity average and R-L difference
surf_aves = []; surf_diffs = []

for dfind in traindf.index:
    velocity, data, isample = get_train_sample(dfind, ftscale=False)
    # velocity, target, values
    (num_vels, y0_velL, y0_velR, y09L_medi, y09R_medi,
         y1029_medi, y3049_medi, y5069_medi, MAE_1medi, MAE_5medi) = info_velocity(
                                            velocity, isample, for_show=False)
    nunique.append(num_vels)
    y0_aves.append((y0_velL + y0_velR)/2); y0_diffs.append(y0_velR - y0_velL)
    y09L_medis.append(y09L_medi); y09R_medis.append(y09R_medi); y1029_medis.append(y1029_medi)
    y3049_medis.append(y3049_medi); y5069_medis.append(y5069_medi)
    MAE_1medis.append(MAE_1medi); MAE_5medis.append(MAE_5medi)
    
    # features are seismic measured values
    velL, velave, velR = info_data(data, isample, for_show=False)
    surf_aves.append(velave)
    surf_diffs.append(velR-velL)

traindf["y_numVels"] = nunique
traindf["y_y0Ave"] = y0_aves
traindf["y_y0Diff"] = y0_diffs
traindf["y_09LMedi"] = y09L_medis
traindf["y_09RMedi"] = y09R_medis
traindf["y_1029Medi"] = y1029_medis
traindf["y_3049Medi"] = y3049_medis
traindf["y_5069Medi"] = y5069_medis
traindf["MAE_1Medi"] = MAE_1medis
traindf["MAE_5Medi"] = MAE_5medis
traindf["x_surfAve"] = surf_aves
traindf["x_surfDiff"] = surf_diffs

# Add color-coding based on the surfDiff and surfAve values
# Red = R-L not zero; Blue = R-L near zero
traindf["diff_clr"] = 'red'
# Use measured R-L difference to set color
seldiff = traindf["x_surfAve"] > (1300.0 + 1200.0*np.log10(1+np.abs(traindf["x_surfDiff"])))
# Use known R-L difference from the target (can't do this for test)
##seldiff = (np.abs(traindf["y_y0Diff"]) < 0.1*diff_color_change)
traindf.loc[seldiff, "diff_clr"] = 'blue'

traindf

# Summary values for the columns
traindf_means = traindf.describe().loc["mean",]
##traindf.describe()

# Some summary information
# Number of discrete velocities:
# 6003 samples have 2 through 16 (except 9)
# 3997 sampes have 41 and above.
np.clip(traindf["y_numVels"],0,40).value_counts()

print("\nAverage MAE wrt the median of each sample: {:.2f}".format(
            traindf_means["MAE_1Medi"]))
print("Average MAE wrt 5 medians in each sample: {:.2f}\n".format(
            traindf_means["MAE_5Medi"]))

# Save the Training Dataframe -- Save it further along after predictions are added.
##traindf.to_csv("traindf.csv", header=True, index=False, float_format='%.2f')

# Velocity Plots from the training dataframe

# For the plots
velocity_range = (1400,4200)


print("\nMedian Ave Surface Velocity: {:.2f}".format(np.median(traindf["x_surfAve"])))
print("Average Ave Surface Velocity: {:.2f}\n".format(np.mean(traindf["x_surfAve"])))

diffs = traindf["x_surfDiff"]

plt.figure(figsize=(8,4))
plt.hist(traindf["x_surfAve"],bins=100)
plt.title("Train: Histogram of the Average Surface Velocity")
plt.xlabel("Surface Velocity (m/s)")
plt.xlim(velocity_range)
plt.savefig("train_hist_surface_velocity.png")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(np.sign(diffs)*np.log10(np.abs(diffs) + 1.0), log=True, bins=100)
plt.title("Train: Histogram of the R-L Velocity Difference")
plt.xlabel("Signed Log10[1+ R-L Surface Velocity Difference (m/s) ]")
plt.savefig("train_hist_velocity_difference.png")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter( np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)), traindf["x_surfAve"],
                             color=traindf["diff_clr"], s=2, alpha=0.25)
lindiffs = np.linspace(-3.0,3.0,100)  # <-- This is log10(1+ abs(surfDiff) )
plt.plot(lindiffs, 1300.0 + 1200.0*np.abs(lindiffs),c='gray',alpha=0.5)
plt.ylabel("Average Surface Velocity (m/s)")
plt.xlabel("Signed Log10[1+ R-L Velocity Difference (m/s) ]")
plt.title("Train: Average Surface Velocity vs. R-L Velocity Difference")
plt.ylim(velocity_range)
plt.savefig("train_scatter_velocity_vs_difference.png")
plt.show()

# Look into the y=0 Average and Difference

# Scatter plot of the y=0 row Average and y=0 R-L Difference velocities
if True:
    diffs = traindf["y_y0Diff"]

    plt.figure(figsize=(6,3))
    plt.scatter( np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)), traindf["y_y0Ave"]/1000,
                             color=traindf["diff_clr"], s=2, alpha=0.25)
    plt.ylabel("Ave y=0 Velocity (km/s)")
    plt.xlabel("Signed Log10[1+ y=0 R-L Velocity Diff (m/s) ]")
    plt.title("Train: y=0 Average Velocity vs. y=0 R-L Velocity Difference")
    plt.savefig("train_y0_scatter_velocity_vs_diff.png")
    plt.ylim(1.4,4.6) # in km/s
    plt.show()

    # Histogra of the y=0 R-L Diff
    plt.figure(figsize=(6,3))
    plt.hist(np.sign(diffs)*np.log10(np.abs(diffs) + 1.0), log=True, bins=100)
    plt.title("Train: Histogram of the y=0 R-L Velocity Difference")
    plt.xlabel("Signed Log10[1+ R-L y=0 Velocity Difference (m/s) ]")
    plt.savefig("train_hist_y0_difference.png")
    plt.show()

    # Scatter plot of the Seismic R-L Diff vs the y=0 R-L Diff
    diffs = traindf["x_surfDiff"]
    diffy0 = traindf["y_y0Diff"]

    plt.figure(figsize=(6,3))
    plt.scatter( np.sign(diffy0)*(np.log10(np.abs(diffy0) + 1.0)),
                    np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)),
                             color=traindf["diff_clr"], s=2, alpha=0.25)
    plt.xlabel("y=0  Log10[1+ R-L Velocity Diff (m/s) ]")
    plt.ylabel("Seismic  Log10[1+ R-L Velocity Diff (m/s) ]")
    plt.title("Train: Seismic R-L Difference vs the y=0 R-L Difference")
    plt.savefig("train_scatter_diff_vs_diff.png")
    plt.show()

# Find some with y=0 diff = 0 and yet seismic R-L is high
##traindf[(traindf["y_y0Diff"] == 0) & (traindf["x_surfDiff"] > 200)]

# Compare measured surface velocity with the y=0 average.
# Include a simple degree 1 polynomial fit 
model = np.poly1d(np.polyfit(np.array(traindf["y_y0Ave"]), 
                             np.array(traindf["x_surfAve"]), 1))
# for polynomial line visualization 
polyline = np.linspace(1400, 4500, 100)  

plt.figure(figsize=(4,4))
plt.scatter( traindf["y_y0Ave"], traindf["x_surfAve"],
                color=traindf["diff_clr"], s=2, alpha=0.25)
plt.plot(polyline, model(polyline), c='orange',alpha=0.6)
plt.xlabel("y=0 Average Velocity")
plt.ylabel("Seismic Average Surface Velocity (m/s)")
plt.title("Train: Seismic Surface Velocity vs. y=0 Velocity")
plt.xlim(1400,4600)
plt.ylim(velocity_range)
plt.savefig("train_surf_vs_y0.png")
plt.show()

print("   Fit coefs [slope, intercept]:", model.coef,"\n")

# Model the Row Ranges Median Velocities
# Look at the velocities in row ranges vs the surface velocity and velocity difference.
# Create simple model fits for each region.

surfAves = traindf["x_surfAve"]
surfDiffs = traindf["x_surfDiff"]
log_surfDiffs = np.sign(surfDiffs)*(np.log10(np.abs(surfDiffs) + 1.0))

# Fit red, blue separately, limit the surfAve range used
selblue = (traindf["diff_clr"] == 'blue') & (traindf["x_surfAve"] < 4100)
selred = (traindf["diff_clr"] == 'red') & (traindf["x_surfAve"] < 4100)

# for polynomial line visualization 
polyline = np.linspace(1400, 4200, 100)

# Save the fit models
rows_models = []
for y_rows in ["09L", "09R", "1029", "3049", "5069"]:
    
    rows_values = traindf["y_"+y_rows+"Medi"]
    surf_values = surfAves.copy()
    vel_axis_label = "Ave Surface Velocity (m/s)"
    degree = 5
    # Modify surf_values for the 09L,R data
    if "09L" in y_rows:
        surf_values = surf_values - 0.5*surfDiffs
        vel_axis_label = "L Surface Velocity (m/s)"
        degree = 3
    if "09R" in y_rows:
        surf_values = surf_values + 0.5*surfDiffs
        vel_axis_label = "R Surface Velocity (m/s)"
        degree = 3

    plt.figure(figsize=(7,4))
    plt.scatter(surf_values, rows_values, color=traindf["diff_clr"], s=2, alpha=0.25)

    # Blue polynomial fit:
    if "09" in y_rows:
        # Use combined L and R data for the model, selblue:
        surf_RLvalues = np.concatenate( ( (surfAves - 0.5*surfDiffs)[selblue], 
                                            (surfAves + 0.5*surfDiffs)[selblue] ) )
        rows_RLvalues = np.concatenate( ( traindf.loc[selblue,"y_09LMedi"], 
                                            traindf.loc[selblue,"y_09RMedi"] ) )
        xmeds, ymeds = xy_medians(surf_RLvalues, rows_RLvalues, 25)
        plt.scatter(xmeds, ymeds, s=8, alpha=1.0, c='darkblue')
        model = np.poly1d(np.polyfit(xmeds, ymeds, degree))
    else:
        xmeds, ymeds = xy_medians(np.array(surf_values[selblue]),
                                    np.array(rows_values[selblue]), 25)
        plt.scatter(xmeds, ymeds, s=8, alpha=1.0, c='darkblue')
        model = np.poly1d(np.polyfit(xmeds, ymeds, degree))
    
    rows_models.append(model)
    blue_resids = (-1.0*model(np.array(surf_values[selblue])) + 
                             np.array(rows_values[selblue]))
    print("  Blue Fit coefs:", model.coef)
    plt.plot(polyline, model(polyline), c='blue',alpha=1.0)
    #
    # Red polynomial fit:
    if "09" in y_rows:
        # Use combined L and R data for the model, selred:
        surf_RLvalues = np.concatenate( ( (surfAves - 0.5*surfDiffs)[selred], 
                                            (surfAves + 0.5*surfDiffs)[selred] ) )
        rows_RLvalues = np.concatenate( ( traindf.loc[selred,"y_09LMedi"], 
                                            traindf.loc[selred,"y_09RMedi"] ) )
        xmeds, ymeds = xy_medians(surf_RLvalues, rows_RLvalues, 25)
        plt.scatter(xmeds, ymeds, s=8, alpha=1.0, c='darkred')
        model = np.poly1d(np.polyfit(xmeds, ymeds, degree))
    else:
        xmeds, ymeds = xy_medians(np.array(surf_values[selred]), 
                                     np.array(rows_values[selred]), 25)
        plt.scatter(xmeds, ymeds, s=8, alpha=1.0, c='darkred')
        model = np.poly1d(np.polyfit(xmeds, ymeds, degree))
    rows_models.append(model)
    red_resids = (-1.0*model(np.array(surf_values[selred])) + 
                             np.array(rows_values[selred]))
    print("  Red Fit coefs:", model.coef)
    plt.plot(polyline, model(polyline), c='purple',alpha=1.0)
    
    plt.xlabel(vel_axis_label)
    plt.xlim(1400, 4200) # reduce because of fitting range
    plt.ylabel("y_"+y_rows+" Median")
    plt.ylim(1400, 4600)
    plt.title("Train: y_"+y_rows+" Median vs. Surface Velocity")
    plt.savefig("train_rows"+y_rows+"_vs_average.png")
    plt.show()


    # Show the residuals vs surface difference for the 09L, 09R
    if "09" in y_rows:
        plt.figure(figsize=(7,2))
        plt.scatter( log_surfDiffs[selblue], blue_resids,
                             color=traindf.loc[selblue,"diff_clr"], s=2, alpha=0.25)
        plt.scatter( log_surfDiffs[selred], red_resids,
                             color=traindf.loc[selred,"diff_clr"], s=2, alpha=0.25)
        plt.ylim(-1000,1000)
        plt.xlabel("Signed Log10[1+ R-L Velocity Diff (m/s) ]")
        plt.ylabel("y_"+y_rows+" Residuals")
        plt.title("Train: y_"+y_rows+" * Residuals * vs. Surface Difference")
        plt.savefig("train_residuals"+y_rows+"_vs_difference.png")
        plt.show()
        
    
    # Show the median values vs surface difference
    plt.figure(figsize=(7,2))
    plt.scatter(log_surfDiffs, rows_values,
                             color=traindf["diff_clr"], s=2, alpha=0.25)
    plt.xlabel("Signed Log10[1+ R-L Velocity Diff (m/s) ]")
    plt.ylabel("y_"+y_rows+" Median")
    plt.ylim(1400, 4600)
    plt.title("Train: y_"+y_rows+" Median vs. Surface Difference")
    plt.savefig("train_rows"+y_rows+"_vs_difference.png")
    plt.show()

    print("\n")

polyline = np.linspace(1400, 4100, 100)  
plt.figure(figsize=(6,3))
for imod in range(5):
    plt.plot(polyline, rows_models[2*imod](polyline), c='blue',alpha=0.6)
    plt.plot(polyline, rows_models[2*imod+1](polyline), c='red',alpha=0.6)
plt.xlabel("Average Surface Velocity (m/s)")
plt.ylabel("Median of Rows")
plt.title("Fits of Row-Ranges Medians vs. Surface Velocity")
plt.show()

# Mystery blue lines re when the region median equals the y=0 velocity.

# What/why are the blue lines in the 1029 and 3049 median vs surface velocity plots?
# Find the samples in these lines
trainblue = traindf[traindf["diff_clr"] == 'blue']

print("\n\n  Look for 'blue' samples that have Rows Medians equal to the y=0 Average.")
print("  - List the counts of Velocity-Map Types.")
print("  - Check the y0Diff values: they are all 0, so vmaps are R-L symmetric.\n\n")

for yrows in ["1029","3049"]:
    plt.figure(figsize=(6,2))
    plt.hist(np.clip(trainblue["y_"+yrows+"Medi"] - trainblue["y_y0Ave"],-800,800),
             log=True, bins=160)
    plt.xlim(-500,500)
    plt.xlabel("Rows "+yrows+" Median  -  y=0 Average")
    plt.show()

    matchdf = trainblue[np.abs(trainblue["y_"+yrows+"Medi"] - trainblue["y_y0Ave"]) < 0.0001]
    print(matchdf["veltype"].value_counts())
    print(matchdf["y_y0Diff"].value_counts())

# Evaluate MAE for the training data
# Compare actual MAE with these values:
print("\nMAE if predicted the median of each sample: {:.2f}".format(
            traindf_means["MAE_1Medi"]))
print("MAE if predicted the 5 row-range medians in each sample: {:.2f}\n".format(
            traindf_means["MAE_5Medi"]))

plt.figure(figsize=(6,3))
plt.hist(traindf["MAE_5Medi"],bins=100)
plt.xlabel("MAE of the sample")
plt.title("Histogram of the MAEs using 5 known medians")
plt.show()

# Add model-predicted columns to the training dataframe based on x_surfAve
# Model order is blue then red for each rows range.
surf_values = traindf["x_surfAve"]
surf_diffs = traindf["x_surfDiff"]
surf_L_values = surf_values - 0.5*surf_diffs
surf_R_values = surf_values + 0.5*surf_diffs
selblue = traindf["diff_clr"] == 'blue'
# Blue and Red model for each row range
for imod, y_rows in enumerate(["09L", "09R", "1029", "3049", "5069"]):
    if y_rows == "09L":
        traindf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_L_values[selblue])
        traindf.loc[-selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_L_values[-selblue])
    elif y_rows == "09R":
        traindf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_R_values[selblue])
        traindf.loc[-selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_R_values[-selblue])
    else:
        traindf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_values[selblue])
        traindf.loc[-selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_values[-selblue])

# Look at the errors in predicting the medians vs the surface average velocity
# These are just the deviations of the points from the model curves in the plots above.
surfAves = traindf["x_surfAve"]
surfDiffs = traindf["x_surfDiff"]
surf_L_values = surfAves - 0.5*surfDiffs
surf_R_values = surfAves + 0.5*surfDiffs
for y_rows in ["09L", "09R", "1029", "3049", "5069"]:
    surf_values = surfAves
    vel_axis_label = "Ave Surface Velocity (m/s)"
    # Modify surf_values for the 09L,R data
    if "09L" in y_rows:
        surf_values = surf_L_values
        vel_axis_label = "L Surface Velocity (m/s)"
    if "09R" in y_rows:
        surf_values = surf_R_values
        vel_axis_label = "R Surface Velocity (m/s)"

    plt.figure(figsize=(6,3))
    plt.scatter(surf_values, traindf["y_"+y_rows+"Medi"] - traindf["pred_"+y_rows],
                        c=traindf["diff_clr"],s=2, alpha=0.15)
    plt.ylim(-1500,1500)
    plt.xlim(1400, 4200)
    plt.title("Error in Predicted Medians for Rows "+y_rows)
    plt.xlabel(vel_axis_label)
    plt.show()

# Calculate the average MAE based on the 5 predicted medians.
# Add MAE_pred to the dataframe for each sample.
MAE_preds = []
for dfind in traindf.index:
    # Read in the data for this sample
    velocity, data, isample = get_train_sample(dfind, ftscale=False)
    # Go through the 5 row regions and calculate MAE wrt their predicted medians
    MAE_5medi = 0.0
    MAE_5medi += 5.0*np.mean(np.abs(velocity[isample,0, 0:10 , 0:34+1  ] - 
                                    traindf.loc[dfind,"pred_09L"]))
    MAE_5medi += 5.0*np.mean(np.abs(velocity[isample,0, 0:10 , 35:  ] - 
                                    traindf.loc[dfind,"pred_09R"]))
    MAE_5medi += 20.0*np.mean(np.abs(velocity[isample,0, 10:29+1 , :  ] - 
                                     traindf.loc[dfind,"pred_1029"]))
    MAE_5medi += 20.0*np.mean(np.abs(velocity[isample,0, 30:49+1 , :  ] - 
                                     traindf.loc[dfind,"pred_3049"]))
    MAE_5medi += 20.0*np.mean(np.abs(velocity[isample,0, 50: , :  ] - 
                                     traindf.loc[dfind,"pred_5069"]))
    MAE_preds.append(MAE_5medi / 70.0)

traindf["MAE_pred"] = MAE_preds

# Save the training dataframe with predictions, etc.
traindf.to_csv("traindf.csv", header=True, index=False, float_format='%.2f')

traindf

print("\nOverall MAE of the predictions is: {:.2f}\n".format(np.mean(traindf["MAE_pred"])))

plt.figure(figsize=(6,3))
plt.hist(traindf["MAE_pred"],bins=100)
plt.xlabel("MAE of the sample")
plt.title("Histogram of the MAEs of the 5-median Predictions")
plt.show()

# Create and fill the test dataframe
# Use the sample submission to get the test ids
submis = pd.read_csv("/kaggle/input/waveform-inversion/sample_submission.csv")

# Create a df of just the test ids (with _y_0)
oiddf = submis.loc[0:4607260:70,["oid_ypos"]].copy()
oiddf = oiddf.reset_index(drop=True)
##oiddf

# For each sample, add the measured surface velocity average and the R-L difference
ave_vels = []
diff_vels = []
for indoid in oiddf.index:
    testdata = np.load('/kaggle/input/waveform-inversion/test/' + 
                   oiddf.loc[indoid,"oid_ypos"][0:10]+'.npy')
    velL, velave, velR = info_data(testdata, for_show=False)
    ave_vels.append(velave)
    diff_vels.append(velR-velL)

oiddf["x_surfAve"] = ave_vels
oiddf["x_surfDiff"] = diff_vels
##oiddf

# Add color-coding based on the surfDiff and surfAve values
oiddf["diff_clr"] = 'red'
# Set blue, same criteria as for the training data
seldiff = oiddf["x_surfAve"] > (1300.0 + 1200.0*np.log10(1+np.abs(oiddf["x_surfDiff"])))
oiddf.loc[seldiff, "diff_clr"] = 'blue'

# Add model-predicted columns to the test dataframe based on x_surfAve
# Order is blue then red model for each rows range.
surf_values = oiddf["x_surfAve"]
surf_diffs = oiddf["x_surfDiff"]
surf_L_values = surf_values - 0.5*surf_diffs
surf_R_values = surf_values + 0.5*surf_diffs
selblue = oiddf["diff_clr"] == 'blue'
# Blue and Red model for each row range
for imod, y_rows in enumerate(["09L", "09R", "1029", "3049", "5069"]):
    if y_rows == "09L":
        oiddf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_L_values[selblue])
        oiddf.loc[-selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_L_values[-selblue])
    elif y_rows == "09R":
        oiddf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_R_values[selblue])
        oiddf.loc[-selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_R_values[-selblue])
    else:
        oiddf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_values[selblue])
        oiddf.loc[-selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_values[-selblue])

oiddf

# Save the Test Dataframe
oiddf.to_csv("oiddf.csv", header=True, index=False, float_format='%.2f')

# Velocity Plots from the Test Dataframe
print("\nMedian Ave Surface Velocity: {:.2f}".format(np.median(oiddf["x_surfAve"])))
print("Average Ave Surface Velocity: {:.2f}\n".format(np.mean(oiddf["x_surfAve"])))

plt.figure(figsize=(8,4))
plt.hist(oiddf["x_surfAve"],bins=100)
plt.title("Test: Histogram of the Average Surface Velocity")
plt.xlabel("Surface Velocity (m/s)")
plt.xlim(velocity_range)
plt.savefig("test_hist_surface_velocity.png")
plt.show()

plt.figure(figsize=(8,4))
diffs = oiddf["x_surfDiff"]
plt.hist(np.sign(diffs)*np.log10(np.abs(diffs) + 1.0), log=True, bins=100)
plt.title("Test: Histogram of the R-L Velocity Difference")
plt.xlabel("Signed Log10[1+ R-L Surface Velocity Difference (m/s) ]")
plt.savefig("test_hist_velocity_difference.png")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter( np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)), oiddf["x_surfAve"],
                             color=oiddf["diff_clr"], s=2, alpha=0.15)
plt.ylabel("Ave Surface Velocity (m/s)")
plt.xlabel("Signed Log10[1+ R-L Velocity Diff (m/s) ]")
plt.title("Test: Average Surface Velocity vs R-L Velocity Difference")
plt.ylim(velocity_range)
plt.savefig("test_scatter_velocity_vs_difference.png")
plt.show()


# Submit a Prediction
# Enter the predictions into the submis dataframe
# The predictions are 3 values for each of the 65818 test samples:
# pred_09L value --> rows 0-9
# pred_09R value --> rows 0-9
# pred_1029 value --> rows 10-29
# pred_3049 value --> rows 30-49
# pred_5069 value --> rows 50-69

# For each range of rows,
# fill all 35 x_j values of the 65818 y_i values with the 65818 predicted values.
all_xs = list(submis.columns[1:])
left_xs = list(submis.columns[1:17+1])
right_xs = list(submis.columns[18:])

# Loop over each set of y_i rows and set them equal to the corresponding predicted values

len_oiddf = len(oiddf)

# Rows 0-9, with values adjusted for L (1,3,...,33) and R (35,37,...,69) halves.
fill_values = (np.ones([17,len_oiddf]) * np.array(oiddf["pred_09L"])).T
for iy in range(10):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, left_xs] = fill_values
fill_values = (np.ones([18,len_oiddf]) * np.array(oiddf["pred_09R"])).T
for iy in range(10):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, right_xs] = fill_values


# Rows 10-29
fill_values = (np.ones([35,len_oiddf]) * np.array(oiddf["pred_1029"])).T
for iy in range(10,29+1):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, all_xs] = fill_values
    
# Rows 30-49
fill_values = (np.ones([35,len_oiddf]) * np.array(oiddf["pred_3049"])).T
for iy in range(30,49+1):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, all_xs] = fill_values

# Rows 50-69
fill_values = (np.ones([35,len_oiddf]) * np.array(oiddf["pred_5069"])).T
for iy in range(50,69+1):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, all_xs] = fill_values

submis

# Generate the submission file
submis.to_csv("submission.csv", header=True, index=False, float_format='%.0f')

# Check it
##!ls -s submission.csv
##!tail -5 submission.csv

# Submissions, fyi
#  v4  1980.7    Predicted 1000 for all (so average is 2980.7)
#       710.5    Submit the sample submission
#  v5   605.5    Predicted 2781.0 for y=0-34 and 3781.0 for y=35-69
# v11   -----    Made the Image: Add all 5 sources with surface pulses zeroed - not useful!
# v13   551.4    Predicted 2481.0 for y=0-34 and 3481.0 for y=35-69. +300 not a good idea :)
# v14   543.6    Predict 1966 (median surface vel) in rows 0-9, 2481 in rows 10-39, 3481 in rows 40-69
# v15   495.1    Enter the actual surface velocity into rows 0-9, rows 10-39 = 2481, rows 40-69 = 3481
# v19   493.5    Separate the rows 0-9 predictions into L and R values.
# v20   489.0    Add x1.03 for rows 09 L,R, and use train average medians for 1039 and 4069.
# v21   488.7    Use x1.017 for rows 09 L,R, same otherwise
# v22   489.5    Use the pred_09 model values for rows 09 L,R, same otherwise.
# v24   491.3    Use pred_09 but same for L,R (pred_09 is based on all xs)
# v25   468.2    Filled the submission with the predicted values for each sample in each row range.
#                Changed the seismic velocity calculation to be more accurate. Put back y0 vs surfAve plot.
# v27   475.3    Has y09 model linear, but strange fit.
# v28   484.1    Changed the y09 model to quadratic - strange fit also. Otherwise v27 with some cleaning up. 
# v29   473.1    Set y09 model to a constant.
# v30   478.7    *** Use means in row regions instead of the medians ***
# v33            Back to medians. Added a third predicted layer.
#       442.1    Adjustments to surface velocity meas.: ignore peaks that are too far in time.
# v34   440.3    Use surfDiff to adjust the L and R halves of the y09 predictions.
# v35            Added some words.
# v36   441.6    Better to plot medians directly vs surface velocity (not the ratios.)
#                More adjustments to the surface velocity determination - it's tricky  :)
#                Try to understand the "blue lines" in the medians vs surface velocity plots.
# v37   435.3    Make separate models for the blue (R-L near 0) and red samples, not so different.
# v38   435.1    Add surfAve > 3000 to the 'blue' samples; use degree 3 for model fits.
# v39   434.9    Both fits are degree=3 (v38 was blue degree=3 but red degree=2)
# v40   434.5    Redefined/Improved red/blue determination after comparing to the y0 R-L=0 truth.
# v41   434.1    Reduced fitting range to surfAve < 4000. Changed plot ranges.
# v43   --4.2    Fit 09L and 09R separately and adjust surfDiff factors added on.
# v44   --6.0    v43 wasn't what I intended, try again. yeesh, v44 had double adjustments.
# v45   434.3    Hmmm... Made slight tweak to 09L,R models...
# v46   434.4    Still messing with the 09 stuff
# v47   --4.4    Fit combined 09L,09R so they use same models, equivalent to adding flipped data.
# v48   434.3    Fixed tiny error in v47.
# v50   427.4    Used xy_medians() to transform x,y before the model fitting ~ as if using MAE. 
# v51   427.8    Increased fitting degree to 5; surfAve clipped at 4100; other little changes.
# v52            Calculated MAE values: if all 5 medians in each sample were known MAE = 249.07 .
#                Using each sample's 5 predicted medians gives MAE = 432.84 <-- Similar to LB score.







