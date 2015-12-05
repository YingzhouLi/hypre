/*
 * File:          bHYPRE_PreconditionedSolver_Stub.c
 * Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:44 PST
 * Description:   Client-side glue code for bHYPRE.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 756
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_PreconditionedSolver.h"
#include "bHYPRE_PreconditionedSolver_IOR.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include <stddef.h>
#include "SIDL_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "SIDL_Loader.h"
#endif

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

void
bHYPRE_PreconditionedSolver_addRef(
  bHYPRE_PreconditionedSolver self)
{
  (*self->d_epv->f_addRef)(
    self->d_object);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
bHYPRE_PreconditionedSolver_deleteRef(
  bHYPRE_PreconditionedSolver self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
bHYPRE_PreconditionedSolver_isSame(
  bHYPRE_PreconditionedSolver self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

SIDL_BaseInterface
bHYPRE_PreconditionedSolver_queryInt(
  bHYPRE_PreconditionedSolver self,
  const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self->d_object,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
bHYPRE_PreconditionedSolver_isType(
  bHYPRE_PreconditionedSolver self,
  const char* name)
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

SIDL_ClassInfo
bHYPRE_PreconditionedSolver_getClassInfo(
  bHYPRE_PreconditionedSolver self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * Set the MPI Communicator.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetCommunicator(
  bHYPRE_PreconditionedSolver self,
  void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self->d_object,
    mpi_comm);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetIntParameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetDoubleParameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetStringParameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetIntArray1Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArray1Parameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetIntArray2Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArray2Parameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetDoubleArray1Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArray1Parameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetDoubleArray2Parameter(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArray2Parameter)(
    self->d_object,
    name,
    value);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_GetIntValue(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self->d_object,
    name,
    value);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_GetDoubleValue(
  bHYPRE_PreconditionedSolver self,
  const char* name,
  double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self->d_object,
    name,
    value);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_Setup(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Vector b,
  bHYPRE_Vector x)
{
  return (*self->d_epv->f_Setup)(
    self->d_object,
    b,
    x);
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_Apply(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x)
{
  return (*self->d_epv->f_Apply)(
    self->d_object,
    b,
    x);
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetOperator(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Operator A)
{
  return (*self->d_epv->f_SetOperator)(
    self->d_object,
    A);
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetTolerance(
  bHYPRE_PreconditionedSolver self,
  double tolerance)
{
  return (*self->d_epv->f_SetTolerance)(
    self->d_object,
    tolerance);
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetMaxIterations(
  bHYPRE_PreconditionedSolver self,
  int32_t max_iterations)
{
  return (*self->d_epv->f_SetMaxIterations)(
    self->d_object,
    max_iterations);
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetLogging(
  bHYPRE_PreconditionedSolver self,
  int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self->d_object,
    level);
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetPrintLevel(
  bHYPRE_PreconditionedSolver self,
  int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self->d_object,
    level);
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_GetNumIterations(
  bHYPRE_PreconditionedSolver self,
  int32_t* num_iterations)
{
  return (*self->d_epv->f_GetNumIterations)(
    self->d_object,
    num_iterations);
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_GetRelResidualNorm(
  bHYPRE_PreconditionedSolver self,
  double* norm)
{
  return (*self->d_epv->f_GetRelResidualNorm)(
    self->d_object,
    norm);
}

/*
 * Set the preconditioner.
 * 
 */

int32_t
bHYPRE_PreconditionedSolver_SetPreconditioner(
  bHYPRE_PreconditionedSolver self,
  bHYPRE_Solver s)
{
  return (*self->d_epv->f_SetPreconditioner)(
    self->d_object,
    s);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__cast(
  void* obj)
{
  bHYPRE_PreconditionedSolver cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (bHYPRE_PreconditionedSolver) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.PreconditionedSolver");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_PreconditionedSolver__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_createCol(int32_t        dimen,
                                             const int32_t lower[],
                                             const int32_t upper[])
{
  return (struct 
    bHYPRE_PreconditionedSolver__array*)SIDL_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_createRow(int32_t        dimen,
                                             const int32_t lower[],
                                             const int32_t upper[])
{
  return (struct 
    bHYPRE_PreconditionedSolver__array*)SIDL_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_PreconditionedSolver__array*)SIDL_interface__array_create1d(len);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_PreconditionedSolver__array*)SIDL_interface__array_create2dCol(m, n);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_PreconditionedSolver__array*)SIDL_interface__array_create2dRow(m, n);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_borrow(
  bHYPRE_PreconditionedSolver*firstElement,
                                          int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct 
    bHYPRE_PreconditionedSolver__array*)SIDL_interface__array_borrow(
    (struct SIDL_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_smartCopy(struct 
  bHYPRE_PreconditionedSolver__array *array)
{
  return (struct bHYPRE_PreconditionedSolver__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

void
bHYPRE_PreconditionedSolver__array_addRef(struct 
  bHYPRE_PreconditionedSolver__array* array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array *)array);
}

void
bHYPRE_PreconditionedSolver__array_deleteRef(struct 
  bHYPRE_PreconditionedSolver__array* array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array *)array);
}

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get1(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1)
{
  return (bHYPRE_PreconditionedSolver)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get2(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2)
{
  return (bHYPRE_PreconditionedSolver)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get3(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3)
{
  return (bHYPRE_PreconditionedSolver)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get4(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3,
                                        const int32_t i4)
{
  return (bHYPRE_PreconditionedSolver)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_PreconditionedSolver
bHYPRE_PreconditionedSolver__array_get(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                       const int32_t indices[])
{
  return (bHYPRE_PreconditionedSolver)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

void
bHYPRE_PreconditionedSolver__array_set1(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        bHYPRE_PreconditionedSolver const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_PreconditionedSolver__array_set2(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        bHYPRE_PreconditionedSolver const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_PreconditionedSolver__array_set3(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3,
                                        bHYPRE_PreconditionedSolver const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_PreconditionedSolver__array_set4(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                        const int32_t i1,
                                        const int32_t i2,
                                        const int32_t i3,
                                        const int32_t i4,
                                        bHYPRE_PreconditionedSolver const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_PreconditionedSolver__array_set(struct 
  bHYPRE_PreconditionedSolver__array* array,
                                       const int32_t indices[],
                                       bHYPRE_PreconditionedSolver const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

int32_t
bHYPRE_PreconditionedSolver__array_dimen(const struct 
  bHYPRE_PreconditionedSolver__array* array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

int32_t
bHYPRE_PreconditionedSolver__array_lower(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                         const int32_t ind)
{
  return SIDL_interface__array_lower((struct SIDL_interface__array *)array,
    ind);
}

int32_t
bHYPRE_PreconditionedSolver__array_upper(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                         const int32_t ind)
{
  return SIDL_interface__array_upper((struct SIDL_interface__array *)array,
    ind);
}

int32_t
bHYPRE_PreconditionedSolver__array_stride(const struct 
  bHYPRE_PreconditionedSolver__array* array,
                                          const int32_t ind)
{
  return SIDL_interface__array_stride((struct SIDL_interface__array *)array,
    ind);
}

int
bHYPRE_PreconditionedSolver__array_isColumnOrder(const struct 
  bHYPRE_PreconditionedSolver__array* array)
{
  return SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)array);
}

int
bHYPRE_PreconditionedSolver__array_isRowOrder(const struct 
  bHYPRE_PreconditionedSolver__array* array)
{
  return SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)array);
}

void
bHYPRE_PreconditionedSolver__array_copy(const struct 
  bHYPRE_PreconditionedSolver__array* src,
                                              struct 
  bHYPRE_PreconditionedSolver__array* dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array *)src,
                             (struct SIDL_interface__array *)dest);
}

struct bHYPRE_PreconditionedSolver__array*
bHYPRE_PreconditionedSolver__array_ensure(struct 
  bHYPRE_PreconditionedSolver__array* src,
                                          int32_t dimen,
                                          int     ordering)
{
  return (struct bHYPRE_PreconditionedSolver__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}
