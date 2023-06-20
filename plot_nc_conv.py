import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

dir='/home/ab1679/sdc_scripts'
ncfilename='sdc_w1'
fn = dir + '/' + ncfilename + '.nc'

ds = nc.Dataset(fn)

D_errors = ds['error'][:]

dts = ds['dt'][:]

kvals = ds['k'][:]

cols = ['b','g','r','c']

print(D_errors)
print(dts)
print(kvals)
for i in range(len(kvals)):
    plt.loglog(dts, D_errors[i,:], cols[i], label='SDC%s'%(kvals[i]))

plt.legend()
plt.title("Williamson1 D Convergece")
# figname = "sdc_w1_D_conv.png"
# plt.savefig(figname)
plt.show()