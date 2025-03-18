import xarray as xr
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution, spi

ds = xr.open_dataset("download/cfsv2_precip.nc")
prec = ds["prec"][:, 1:8, :, :, :]
shp = prec.shape
spi3 = prec.copy(deep=True)
spi3.name = "spi3"

for i in range(shp[1]):
    for j in range(shp[2]):
        for k in range(shp[3]):
            for l in range(shp[4]):
                print(i, j, k, l)
                prec1 = prec[:, i, j, k, l].values
                spi3[:, i, j, k, l].values = spi(
                    prec1,
                    scale=3,
                    distribution=Distribution.pearson,
                    data_start_year=1982,
                    calibration_year_initial=1982,
                    calibration_year_final=2024,
                    periodicity=Periodicity.monthly,
                )

spi3.to_dataset().to_netcdf("spi3.nc")
