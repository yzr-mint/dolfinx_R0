# Original multiphenics code Copyright (C) 2016-2022 by the multiphenics authors
# Modifications and additional code by yzr-mint, 2024 under the same license.
#
# This file utilizes the multiphenics library, which is covered under the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This file is independent and not part of the multiphenics project. 
# It is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this code. If not, see <http://www.gnu.org/licenses/>.
#
# Modifications made to the multiphenics code:
# - Changed the boundary condition to accommodate non-linear dynamics.
# - Demonstrated how to add a Lagrange multiplier for constrained optimization.
#

import os
os.environ['OMP_NUM_THREADS'] = '1'

import typing
import dolfinx.mesh
import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import gmsh
import mpi4py.MPI
import numpy as np
import numpy.typing
import petsc4py.PETSc as PETSc
import ufl
PETSc.Sys.pushErrorHandler("traceback")

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

PETSc.Log.begin()

L = 5.0
H = 2.0

msh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD,
                    points=((0.0, 0.0), (L, H)),
                    n=(int(L*30), int(H*30)),
                    cell_type=dolfinx.mesh.CellType.triangle, # quadrilateral triangle
                    ghost_mode=dolfinx.mesh.GhostMode.shared_facet
                    )

V_element = basix.ufl.element("CG", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
Q_element = basix.ufl.element("CG", msh.basix_cell(), 1)

def LEFT(x):
    return np.isclose(x[0], 0)
def RIGHT(x):
    return np.isclose(x[0], L)
def WALL(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], H))
def v_inflow_eval(x):
    values = np.zeros((2, x.shape[1]))
    values[0] = 4.0 * x[1] * (H - x[1]) / H / H
    values[1] = np.zeros(x.shape[1])
    return values


W_element = basix.ufl.mixed_element([V_element, Q_element])
W = dolfinx.fem.functionspace(msh, W_element)

vq = ufl.TestFunction(W)
(v, q) = ufl.split(vq)
dup = ufl.TrialFunction(W)
up = dolfinx.fem.Function(W)
(u, p) = ufl.split(up)

Re = 100
V_char = 1.0
L_char = 2.0
nu = dolfinx.fem.Constant(msh, V_char * L_char / Re)

# Variational forms
F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(ufl.div(u), q) * ufl.dx)
J = ufl.derivative(F, up, dup)

# Boundary conditions
boundaries_in = dolfinx.mesh.locate_entities_boundary(msh, 1, LEFT)
boundaries_wall = dolfinx.mesh.locate_entities_boundary(msh, 1, WALL)
boundaries_out = dolfinx.mesh.locate_entities_boundary(msh, 1, RIGHT)

W0 = W.sub(0)
V, _ = W0.collapse()
u_in = dolfinx.fem.Function(V)
u_in.interpolate(v_inflow_eval)

bdofs_V_in = dolfinx.fem.locate_dofs_topological((W0, V), msh.topology.dim - 1, boundaries_in)
bdofs_V_wall = dolfinx.fem.locate_dofs_topological((W0, V), msh.topology.dim - 1, boundaries_wall)
bdofs_V_out = dolfinx.fem.locate_dofs_topological((W0, V), msh.topology.dim - 1, boundaries_out)
inlet_bc = dolfinx.fem.dirichletbc(u_in, bdofs_V_in, W0)
wall_bc = dolfinx.fem.dirichletbc(u_in, bdofs_V_wall, W0)
outlet_bc = dolfinx.fem.dirichletbc(u_in, bdofs_V_out, W0)
bc = [inlet_bc, wall_bc, outlet_bc]

# Class for interfacing with SNES
# https://github.com/multiphenics/multiphenics/blob/master/tutorials/02_navier_stokes/navier_stokes.py
class NavierStokesProblem:
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(
        self, F: ufl.Form, J: ufl.Form, solution: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC], P: typing.Optional[ufl.Form] = None
    ) -> None:
        self._F = dolfinx.fem.form(F)
        self._J = dolfinx.fem.form(J)
        
        self._solution = solution
        self._bcs = bcs
        self._P = P

        self.J_l = dolfinx.fem.form(q * ufl.dx) # lagrange multiplier
        self.c_vec = dolfinx.fem.petsc.assemble_vector(self.J_l)
        self.c_vec.assemble()
        local_size = self.c_vec.getLocalSize()
        self._obj_vec = PETSc.Vec().createMPI((local_size + (1 if comm.rank == comm.size - 1 else 0), PETSc.DECIDE), comm=comm)

    def update_solution(self, x: PETSc.Vec) -> None:
        """Update `self._solution` with data in `x`."""
        is_uh = PETSc.IS().createGeneral(range(*(self._solution.vector.getOwnershipRange())), comm=PETSc.COMM_WORLD)

        result_x = x.getSubVector(is_uh)
        self._solution.vector.setValues(is_uh, result_x)
        self._solution.vector.assemble()
        self._solution.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def obj(
        self, snes: PETSc.SNES, x: PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()


    def F(
        self, snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec
    ) -> None:
        """Assemble the residual."""
        self.update_solution(x)
        F_ori = dolfinx.fem.petsc.assemble_vector(self._F)
        dolfinx.fem.petsc.apply_lifting(F_ori, [self._J], [self._bcs], x0=[self._solution.vector], scale=-1.0)
        F_ori.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F_ori, self._bcs, self._solution.vector, -1.0)

        # compute full residule
        if rank == size - 1:
            lam_l = x.getValue(x.getOwnershipRange()[1] - 1)
        else:
            lam_l = 0
        lam = comm.allreduce(lam_l, op = mpi4py.MPI.SUM)
        F_ori.axpy(lam, self.c_vec)
        F_ori.assemble()
        F_vec.setValues(range(*(F_ori.getOwnershipRange())), F_ori)

        cx = self.c_vec.dot(self._solution.vector)
        if comm.rank == comm.size - 1:
            F_vec.setValue(F_ori.getSize(), cx)

        F_vec.assemble()


    @staticmethod
    def gather_to_the_last_process(b):
        # b -> y
        global_rows = b.getSize()
        if comm.rank == comm.size - 1:
            y = PETSc.Vec().createSeq(global_rows, comm=PETSc.COMM_SELF)
            send_index_set = PETSc.IS().createStride(global_rows, first=0, step=1, comm=PETSc.COMM_SELF)
            recv_index_set = PETSc.IS().createStride(global_rows, first=0, step=1, comm=PETSc.COMM_SELF)
        else:
            y = PETSc.Vec().createSeq(0, comm=PETSc.COMM_SELF)
            send_index_set = PETSc.IS().createStride(0, first=0, step=1, comm=PETSc.COMM_SELF)
            recv_index_set = PETSc.IS().createStride(0, first=0, step=1, comm=PETSc.COMM_SELF)
        scatter = PETSc.Scatter().create(b, send_index_set, y, recv_index_set)
        scatter.scatter(b, y, mode=PETSc.ScatterMode.FORWARD, addv=PETSc.InsertMode.INSERT_VALUES)
        return y

    def J(
        self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat,
        P_mat: PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        J_ori = dolfinx.fem.petsc.assemble_matrix(self._J, self._bcs, diagonal=1.0)  # type: ignore[arg-type]

        J_ori.assemble()
        local_size = self.c_vec.getLocalSize()
        global_size = self.c_vec.getSize()
        local_range = range(*(self.c_vec.getOwnershipRange()))
        global_range = range(global_size)

        global_is = PETSc.IS().createGeneral(global_range, comm=PETSc.COMM_WORLD)
        local_is = PETSc.IS().createGeneral(local_range, comm=PETSc.COMM_WORLD)

        # append value of the vector
        J_mat.setValues(local_is, global_size, self.c_vec.getArray(), addv=PETSc.InsertMode.INSERT_VALUES)
        y = self.gather_to_the_last_process(self.c_vec)
        if rank == comm.size - 1:
            J_mat.setValues(global_size, global_is, y.getArray(), addv=PETSc.InsertMode.INSERT_VALUES)
            J_mat.setValues(global_size, global_size, 0, addv=PETSc.InsertMode.INSERT_VALUES)
            
        # copy to a larger matrix
        values = J_ori.getValuesCSR()
        if rank == comm.size - 1:
            values = (np.append(values[0], values[0][-1]), ) + values[1:]
        J_mat.setValuesCSR(*values, addv=PETSc.InsertMode.INSERT_VALUES)


        J_mat.assemble()


# Create problem
problem = NavierStokesProblem(F, J, up, bc)
local_size = problem.c_vec.getLocalSize()
J_aug = PETSc.Mat().createAIJ(((local_size + (1 if comm.rank == comm.size - 1 else 0), PETSc.DECIDE), (local_size + (1 if comm.rank == comm.size - 1 else 0), PETSc.DECIDE)), comm=comm)
F_aug = PETSc.Vec().createMPI((local_size + (1 if comm.rank == comm.size - 1 else 0), PETSc.DECIDE), comm=comm)


# Solve
snes = PETSc.SNES().create(msh.comm)
snes.setTolerances(max_it=20)
snes.getKSP().setType("preonly")
snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")
snes.setObjective(problem.obj)
snes.setFunction(problem.F, F_aug)
snes.setJacobian(problem.J, J=J_aug, P=None)
snes.setMonitor(lambda _, it, residual: print(it, residual))
up_copy = F_aug.duplicate()
snes.solve(None, up_copy)
problem.update_solution(up_copy)
up_copy.destroy()
snes.destroy()


u, p = up.split()
uh = dolfinx.fem.Function(W.sub(0).collapse()[0])
ph = dolfinx.fem.Function(W.sub(1).collapse()[0])
uh.interpolate(u)
ph.interpolate(p)
file_affix = "/home/yzhaorong/dolfin_pctools/dolfin_test/PETScConcat/"
from dolfinx.io import VTXWriter
vtx_u = VTXWriter(msh.comm, file_affix + "dfg2D-3-u.bp", [uh], engine="BP4")
vtx_p = VTXWriter(msh.comm, file_affix + "dfg2D-3-p.bp", [ph], engine="BP4")
vtx_u.write(0)
vtx_p.write(0)
vtx_u.close()
vtx_p.close()

