cdef class Matrix:
    """
    `Matrix` : Class for creating and handling AMGX Matrix objects.

    Examples
    --------

    Uploading the matrix ``[[1, 2], [3, 4]]`` using the `upload` method:

    >>> import pyamgx, numpy as np
    >>> pyamgx.initialize()
    >>> cfg = pyamgx.Config().create("")
    >>> rsrc = pyamgx.Resources().create_simple(cfg)
    >>> M = pyamgx.Matrix().create(rsrc)
    >>> M.upload(
    ...     row_ptrs=np.array([0, 2, 4], dtype=np.int32),
    ...     col_indices=np.array([0, 1, 0, 1], dtype=np.int32),
    ...     data=np.array([1., 2., 3., 4.], dtype=np.float64))
    >>> M.destroy()
    >>> rsrc.destroy()
    >>> cfg.destroy()
    >>> pyamgx.finalize()

    """
    cdef AMGX_matrix_handle mtx

    def create(self, Resources rsrc, mode='dDDI'):
        """
        M.create(Resources rsrc, mode='dDDI')

        Create the underlying AMGX Matrix object.

        Parameters
        ----------
        rsrc : Resources

        mode : str, optional
            String representing data modes to use.

        Returns
        -------
        self : Matrix
        """
        check_error(AMGX_matrix_create(&self.mtx, rsrc.rsrc, asMode(mode)))
        return self

    def comm_from_maps(self, int allocated_halo_depth, int num_import_rings,
                       int max_num_neighbors, int[:] neighbors,
                       int[:] send_ptrs, int[:] send_maps,
                       int[:] recv_ptrs, int[:] recv_maps):
        check_error(AMGX_matrix_comm_from_maps(self.mtx,
                                               allocated_halo_depth, num_import_rings,
                                               max_num_neighbors, &neighbors[0],
                                               &send_ptrs[0], &send_maps[0],
                                               &recv_ptrs[0], &recv_maps[0]))
        return self

    def upload(self, int[:] row_ptrs, int[:] col_indices,
               double[:] data, block_dims=[1, 1]):
        """
        M.upload(row_ptrs, col_indices, data, block_dims=[1, 1])

        Copy data from arrays describing the sparse matrix to
        the Matrix object.

        Parameters
        ----------
        row_ptrs : array_like
            Array of row pointers. For a description of the arrays
            `row_ptrs`, `col_indices` and `data`,
            see `here <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_.
        col_indices : array_like
            Array of column indices.
        data : array_like
            Array of matrix data.
        block_dims : tuple_like, optional
            Dimensions of block in x- and y- directions. Currently
            only square blocks are supported, so block_dims[0] should be
            equal to block_dims[1].

        Returns
        -------
        self : Matrix
        """
        cdef int block_dimx, block_dimy

        block_dimx = block_dims[0]
        block_dimy = block_dims[1]

        nnz = len(data)
        nrows = len(row_ptrs) - 1
        ncols = max(col_indices) + 1

        if nrows != ncols:
            raise ValueError, "Matrix is not square, has shape ({}, {})".format(nrows, ncols)

        check_error(AMGX_matrix_upload_all(
            self.mtx,
            nrows, nnz, block_dimx, block_dimy,
            &row_ptrs[0], &col_indices[0],
            &data[0], NULL))

        return self

    def upload_global(self, int[:] row_ptrs, int[:] col_indices,
                      double[:] data, int halo_depth=1, block_dims=[1, 1],
                      int[:] partition_vector=None):
        cdef int block_dimx, block_dimy
        cdef int *pv
        pv = NULL if partition_vector is None else &partition_vector[0]

        block_dimx = block_dims[0]
        block_dimy = block_dims[1]

        nnz = len(data)
        nrows = len(row_ptrs) - 1
        ncols = max(col_indices) + 1

        if nrows != ncols:
            raise ValueError, "Matrix is not square, has shape ({}, {})".format(nrows, ncols)

        check_error(AMGX_matrix_upload_all_global(
            self.mtx,
            nrows, nrows, nnz, block_dimx, block_dimy,
            &row_ptrs[0], &col_indices[0],
            &data[0], NULL,
            halo_depth, halo_depth, NULL))

        return self

    def upload_CSR(self, csr):
        """
        M.upload_CSR(csr)

        Copy data from a :class:`scipy.sparse.csr_matrix` to the Matrix object.

        Parameters
        ----------
        csr : scipy.sparse.csr_matrix

        Returns
        -------
        self : Matrix
        """
        nrows = csr.shape[0]
        ncols = csr.shape[1]

        if nrows != ncols:
            raise ValueError, "Matrix is not square, has shape ({}, {})".format(nrows, ncols)

        row_ptrs = csr.indptr
        col_indices = csr.indices
        data = csr.data

        if len(col_indices) == 0:
            # assume matrix of zeros
            col_indices = np.array([ncols-1], dtype=np.int32)
            data = np.array([0], dtype=np.float64)

        self.upload(row_ptrs, col_indices, data)
        return self

    def get_size(self):
        """
        M.get_size()

        Get the matrix size (in block units), and the block dimensions.

        Returns
        -------

        n : int
            The matrix size (number of rows/columns) in block units.
        block_dims : tuple
            A tuple (`bx`, `by`) representing the size of the
            blocks in the x- and y- dimensions.
        """
        cdef int n, bx, by
        check_error(AMGX_matrix_get_size(
            self.mtx,
            &n, &bx, &by))
        return n, (bx, by)

    def get_nnz(self):
        """
        M.get_nnz()

        Get the number of non-zero entries of the Matrix.

        Returns
        -------
        nnz : int
        """
        cdef int nnz
        check_error(AMGX_matrix_get_nnz(
            self.mtx,
            &nnz))
        return nnz

    def replace_coefficients(self, double[:] data):
        """
        M.replace_coefficients(data)
        Replace matrix coefficients without changing the nonzero structure.

        Parameters
        ----------
        data : array_like
            Array of matrix data.
        """
        cdef int n, nnz
        size, (bx, by) = self.get_size()
        n = self.get_size()[0]
        nnz = self.get_nnz()
        check_error(AMGX_matrix_replace_coefficients(
            self.mtx, n, nnz, &data[0], NULL))

    def destroy(self):
        """
        M.destroy()

        Destroy the underlying AMGX Matrix object.
        """
        check_error(AMGX_matrix_destroy(self.mtx))
