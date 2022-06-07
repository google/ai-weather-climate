"""Beam pipeline to combine multiple netCDF files into a single Zarr volume.

Assumptions:
  - The netCDF files have the same coords for the coords that exist.
  - There is one dimension (usually 'time') that we want to chunk.
  - There may be some "singleton" dimensions, meaning that the dimension
    is split across netCDF files. For example, we might have separate
    netCDF files for each pressure level.

The implementation takes multiple passes:
  - Compute the volume metadata, and write it to the zarr volume. To do this,
    first read the metadata from each netCDF file, truncate it to the first
    chunk, combine with xarray.merge, and then extend the chunked dimension
    back to full size.
  - Once the metadata is written, then each netCDF file will correspond to
    some exact number of chunks, and it can be written to the final
    Zarr volume.
"""
import contextlib
import functools
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Generator, List, MutableMapping, Optional, Sequence, Text, TypeVar

from absl import logging
import apache_beam as beam
import pandas as pd
import tensorflow as tf
import xarray


@contextlib.contextmanager
def _mkdtemp(**keyword_params) -> Generator[Text, None, None]:
  """Create a local directory, removing it when the operation is complete.

  Args:
    **keyword_params: keyword params to be passed to tempfile.mkdtemp().

  Yields:
    Filename of the temporary file.
  """

  local_dirname = tempfile.mkdtemp(**keyword_params)
  try:
    yield local_dirname
  finally:
    shutil.rmtree(local_dirname)


@contextlib.contextmanager
def _copy_to_temp(source: Text) -> Generator[Text, None, None]:
  """Copies the source file to a temporary directory.

  Args:
    source: The source file to copy.

  Yields:
    The full path to the file in the temporary directory.
  """
  with _mkdtemp() as temp_dir:
    temp_path = os.path.join(temp_dir, os.path.basename(source))
    tf.io.gfile.copy(source, temp_path, overwrite=True)
    yield temp_path


@contextlib.contextmanager
def _open_netcdf(path, *, decode_cf=True):
  """Open a netCDF file from a remote path."""
  with _copy_to_temp(path) as local_path:
    ds = xarray.open_dataset(local_path, decode_cf=decode_cf)
    yield ds


def _extract_index(path: str,
                   chunk_dim: str,
                   chunk_size: int,
                   sel: Optional[Dict[str, Any]] = None) -> pd.Index:
  """Extract the full index for the chunk dimension."""
  logging.info('Fetching index for "%s" from %s', chunk_dim, path)
  with _open_netcdf(path) as ds:
    if sel:
      ds = ds.sel(sel)
    index = ds.indexes[chunk_dim]
  logging.info('Index fetched')
  assert index.size % chunk_size == 0, (
      f'Chunk dimension {chunk_dim} has size {index.size}, '
      f'which is not a multiple of the chunk_size {chunk_size}')
  return index


def _extract_template(path: str,
                      chunk_dim: str,
                      chunk_size: int,
                      sel: Optional[Dict[str, Any]] = None) -> xarray.Dataset:
  """Extract a dummy Dataset with metadata matching a dataset on disk."""
  with _open_netcdf(path) as full_ds:
    if sel:
      full_ds = full_ds.sel(sel)
    if chunk_dim in full_ds.dims:
      # Only include the first chunk worth of data.
      sample = full_ds.head({chunk_dim: chunk_size}).compute()
      # Replace dependent variables with dummies of all zeros using dask:
      # - This ensures the dataset remains small when serialized with pickle
      #   when passed between Beam stages.
      # - This ensures these variables won't get written to disk by to_zarr
      #   with compute=False.
      template = xarray.zeros_like(sample.chunk())
    else:
      template = full_ds.compute()
  return template


def _expand_chunked_dimension(dataset: xarray.Dataset, chunk_dim: str,
                              index: pd.Index,
                              singleton_dims: List[str]) -> xarray.Dataset:
  """Expand an xarray.Dataset to use a new array of times."""
  old_size = dataset.sizes[chunk_dim]
  repeats, remainder = divmod(index.size, old_size)
  if remainder:
    raise ValueError(
        f'new size for dimension "{chunk_dim}" must be a multiple of the old '
        f'size: {index.size} vs {old_size} '
        f'with time:\n{index}\nand dataset:\n{dataset}')

  expanded = xarray.concat(
      [dataset] * repeats, dim=chunk_dim, data_vars='minimal')
  expanded.coords[chunk_dim] = index
  expanded = expanded.chunk({dim: 1 for dim in singleton_dims})
  return expanded


def _write_template_to_zarr(templates: Sequence[xarray.Dataset], chunk_dim: str,
                            index: pd.Index, singleton_dims: List[str],
                            zarr_store: MutableMapping[str, Any]) -> None:
  """Create an empty Zarr file matching the given templates."""
  merged = xarray.merge(templates)
  expanded = _expand_chunked_dimension(merged, chunk_dim, index, singleton_dims)
  # compute=False means don't write data saved in dask arrays
  expanded.to_zarr(zarr_store, compute=False, consolidated=True, mode='w')


def _get_reference_region(keys: List[str], ref_indexes: Dict[str, pd.Index],
                          src_indexes: Dict[str, pd.Index]) -> Dict[str, slice]:
  """Get the region spanned by the keys."""
  region = {}
  for key in keys:
    ref_range = ref_indexes[key]
    src_range = src_indexes[key]
    region[key] = slice(
        ref_range.get_loc(src_range[0]),
        ref_range.get_loc(src_range[-1]) + 1)
  return region


def _copy_netcdf_to_zarr_region(netcdf_path: str,
                                zarr_store: MutableMapping[str, Any],
                                chunk_dim: str,
                                chunk_size: int,
                                singleton_dims: List[str],
                                sel: Optional[Dict[str, Any]] = None):
  """Copy a netCDF file into a Zarr file."""
  reference = xarray.open_zarr(zarr_store, chunks=False)

  with _open_netcdf(netcdf_path) as source_ds:
    if sel:
      source_ds = source_ds.sel(sel)

    # Get the region spanned by the file.
    dims = [chunk_dim]
    chunk_spec = {chunk_dim: chunk_size}
    for dim in singleton_dims:
      if dim in source_ds.dims:
        dims.append(dim)
        chunk_spec[dim] = 1
    region = _get_reference_region(dims, reference.indexes, source_ds.indexes)

    # coordinates were already written as part of the template
    source_ds = source_ds.drop_vars(
        list(source_ds.dims) + list(source_ds.coords))
    source_ds = source_ds.chunk(chunk_spec)

    delayed = source_ds.to_zarr(zarr_store, region=region, compute=False)
    # use multiple threads
    delayed.compute(num_workers=16)


F = TypeVar('F', bound=Callable)


def _counted_evals(func: F, num_expected: int) -> F:
  """Wrap a function such that a Beam Counter counts evaluations."""

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    beam.metrics.Metrics.counter(func.__name__,
                                 f'started-of-{num_expected}').inc()
    result = func(*args, **kwargs)
    beam.metrics.Metrics.counter(func.__name__,
                                 f'finished-of-{num_expected}').inc()
    return result

  return wrapper


class NetcdfToZarr(beam.PTransform):
  """Write a collection of netCDF files to a single Zarr volume.

  Supports a dimension to be chunked (usually time), as well as "singleton"
  dimensions, where each netCDF file contains at most one index in that
  dimension.
  """

  def __init__(self,
               input_paths: List[str],
               chunk_dim: str,
               chunk_size: int,
               singleton_dims: List[str],
               zarr_store: MutableMapping[str, Any],
               sel: Optional[Dict[str, Any]] = None):
    """Pipeline to write a single Zarr volume from a set of netCDF files.

    Args:
      input_paths: names of the netCDF files
      chunk_dim: dimension to be chunked
      chunk_size: size of chunks in the chunk_dim
      singleton_dims: dimensions that are split across netCDF files
      zarr_store: output volume.
      sel: dictionary specifying selection from netCDF files (default None)
    """
    self.input_paths = input_paths
    self.chunk_dim = chunk_dim
    self.chunk_size = chunk_size
    self.singleton_dims = singleton_dims
    self.zarr_store = zarr_store
    self.sel = sel

  def expand(self, p: beam.PTransform) -> beam.PTransform:
    get_index = (
        p
        | 'index path' >> beam.Create(self.input_paths[0:1])
        | 'extract index' >> beam.Map(
            _extract_index,
            chunk_dim=self.chunk_dim,
            chunk_size=self.chunk_size,
            sel=self.sel))

    write_zarr_metadata = (
        p
        | 'example paths' >> beam.Create(self.input_paths)
        | 'extract templates' >> beam.Map(
            _counted_evals(_extract_template, len(self.input_paths)),
            chunk_dim=self.chunk_dim,
            chunk_size=self.chunk_size,
            sel=self.sel)
        | 'combine templates' >> beam.combiners.ToList()
        | 'write zarr template' >> beam.Map(
            _write_template_to_zarr,
            chunk_dim=self.chunk_dim,
            index=beam.pvalue.AsSingleton(get_index),
            singleton_dims=self.singleton_dims,
            zarr_store=self.zarr_store))

    write_zarr_chunks = (
        p
        | 'input paths' >> beam.Create(self.input_paths)
        | 'wait on metadata' >> beam.Map(
            lambda path, _: path, beam.pvalue.AsSingleton(write_zarr_metadata))
        | 'write array data' >> beam.Map(
            _counted_evals(_copy_netcdf_to_zarr_region, len(self.input_paths)),
            zarr_store=self.zarr_store,
            chunk_dim=self.chunk_dim,
            chunk_size=self.chunk_size,
            singleton_dims=self.singleton_dims,
            sel=self.sel))

    return write_zarr_chunks
