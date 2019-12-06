cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

cdef class Resources:
    """
    Resources: Class for creating and freeing AMGX Resources objects.
    """
    cdef AMGX_resources_handle rsrc

    def create(self, Config cfg, MPI.Comm comm, int device_num, int[:] devices):
        comm_ptr = libmpi.MPI_COMM_NULL if comm is None else comm.ob_mpi
        check_error(AMGX_resources_create(&self.rsrc, cfg.cfg, comm_ptr, device_num, &devices[0]))
        return self

    def create_simple(self, Config cfg):
        """
        rsc.create_simple(cfg)

        Create the underlying AMGX Resources object in a
        single-threaded application.

        Parameters
        ----------
        cfg : Config

        Returns
        -------
        self : Resources
        """
        check_error(AMGX_resources_create_simple(&self.rsrc, cfg.cfg))
        return self

    def destroy(self):
        """
        rsc.destroy()

        Destroy the underlying AMGX Resources object.
        """
        check_error(AMGX_resources_destroy(self.rsrc))
