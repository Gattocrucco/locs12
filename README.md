# locs12

Code to study the localization of S1 and S2 signals in DarkSide20k.

## Alphabetical file index by category

### Data

  * `plot_saturation.root`: temporal and spatial distribution of S2 hits.

### Scripts

  * `plot_saturation.py`: plot the contents of `plot_saturation.root`.

  * `temps1plots20210131.py`: save performance and diagnostic plots of s1
    temporal localization filters with the version of `temps1.py` from January
    31, 2021.

  * `temps1plots20210203.py`: save performance and diagnostic plots of s1
    temporal localization filters with the version of `temps1.py` from February
    3, 2021.

All the following `temps1series*.py` scripts write the results to
`temps1series*.npy` and have a companion script `temps1series*plot.py` to do
the plots, which can be used while the main script is still running to show
partial results. The digits in the name are month-day.

  * `temps1series0203.py`: **(OUTDATED)** efficiency vs. number of photons.
    
  * `temps1series0213.py`: ER/NR discrimination and KDE bandwidth.
    
  * `temps1series0214.py`: fast/slow discrimination and KDE bandwidth.
    
  * `temps1series0222.py`: **(OUTDATED)** compare likelihood, cross
    correlation, and coincidence.
    
  * `temps1series0224.py`: **(OUTDATED)** find optimal coincidence time.
    
  * `temps1series02240.py`: **(OUTDATED)** compare likelihood, cross
    correlation, and coincidence.

  * `temps1series0226.py`: find optimal coincidence time.
  
  * `temps1series0226z.py`: find optimal cross correlation template sigma.
  
  * `temps1series0227.py`: compare likelihood, cross correlation, and
    coincidence.
    
### Modules

  * `aligntwin.py`: code to align the ticks of multiple plot scales.

  * `ccdelta.py`: compute the cross-correlation of dicrete points with a
    continuous function.

  * `clusterargsort.py`: filter away values which are close to an higher value
    in a signal.
    
  * `coincth.py`: formulas for the coincidence rate.

  * `dcr.py`: generate uniform hits.
  
  * `downcast.py`: downcast numpy data types recursively.
  
  * `filters.py`: filters to be applied to a temporal sequence of hits.
  
  * `named_cartesian_product.py`: cartesian product of arrays.
  
  * `npzload.py`: class to serialize objects to numpy archives.
    
  * `numba_scipy_special/`: module to add support for `scipy.special` functions 
    in numba.

  * `pS1.py`: **(DEPRECATED)** compute and sample the temporal distribution of
    S1 photons.
  
  * `ps12.py`: compute and sample the temporal distribution of S1 and S2
    photons.
  
  * `qsigma.py`: equivalent of standard deviation with quantiles.
  
  * `runsliced.py`: do something in batches with a progressbar.
  
  * `sampling_bounds.py`: bounds for random ramples.
  
  * `symloglocator.py`: class to place minor ticks on symlog scales.
  
  * `temps1.py`: simulate the temporal localization of S1 signals.
  
  * `testccfilter.py`: class to study where to evaluate the cross correlation
    filter.
  
  * `textbox.py`: draw a box with text on a plot.

## Dependencies

Should work with Python >= 3.6 and the standard Python scientific stack. Just
in case: developed on Python 3.8.2, required modules with version numbers are
listed in `requirements.txt`.
