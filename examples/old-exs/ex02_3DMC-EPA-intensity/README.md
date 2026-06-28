# Ex02 MC Simulation in three dimensional Parameter Space
\
This example ***<font color='red'>ex02_3DMC-EPA-intensity</font>***  gives example outputs from three dimensional MC simulation.
This simulation is done for `EPA` model under ``intensity`` concept. Five different
atomic coordinates are generated as an example. Each coordinate are processed using
reflection order (RO) upto 9.

``pnew`` is base name given to the output files and the number in ***<font color='red'>_xxx</font>*** gives the
RO. Since intersection needs atleast two RO, the file names starts from 2.

The given results files, in python `hdf (version-5)` file format, can be post-processed using the
plot routine given in`` utils/plot3DMC_results.py``. The expected output is given in
``EPA_I.pdf`` file.

## 1. Plotting MC results
\
To plot the results, open your favorite python editor, for example jupyter notebook,
and run the following code


```py
from psc.utils.plot3DMC_results import plot3dmcresults
h = 9

fpath = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop','ex02') 

plot3dmcresults(h, fpath)
```


NOTE: change the 'fpath' variable according to your setting. Current setting assume the
'ex02' folder is in your Desktop location.


Following the same map, you can generate any number of coordinates and can process then using any number
of ROs in 3D PS. Also the 'utils/plot3DMC_results.py' routine can handle these results. 

NOTE: the routine 'utils/plot3DMC_results.py' is meant for only 3D MC simulation.

## 2. Time plot

you can visualize time taken for the process using plottime_nD module from plottime as

```py
from psc.utils.plottime  import plottime_nD

fpath = os.path.join(os.getcwd(), 'psc\\examples\\ex02')
h = 9 ; 
plottime_nD(fpath, h, interval=1)

```

the module ``plottime_nD`` need ``fpath`` to search for the results (in h5 format), total number of ROs h used in the simulation and the desired interval. This save the time analysis in ``timeinfo.pdf`` file in the path defined by the variable ``fpath``.

For the ex02 the time analysis is save in ``timeinfo.pdf`` file. The analysis is done with ``interval=2``.

## 3. How to run
\
See the provided  ``ex02.ipjnb`` jupyter notebook. Execute all cells