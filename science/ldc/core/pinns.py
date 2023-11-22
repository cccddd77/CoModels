# Copyright (c) 2023 OneFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import numpy as np
import oneflow as flow
from .utils import cfg, sample, cache_func, check_indexes, tensor, save_checkpoint, load_checkpoint, save_vtk


class Domain(object):

    def __init__(self, dim, time_dependent, discreted):
        self.dim = dim
        self.time_dependent = time_dependent
        self.discreted = discreted
        self.dtype = cfg.get_dtype_n()

    def get_dim(self):
        return self.dim

    def is_discreted(self):
        return self.discreted

    def is_time_dependent(self):
        return self.time_dependent

    def get_dtype(self):
        return self.dtype

    def get_interior_points(self, time=None):
        if not self.discreted:
            raise ValueError(
                "This interface is only valid within dicreted domain.")
        if time is not None and not self.time_dependent:
            raise ValueError(
                "Try get points at specific time on a time independent geometry domain."
            )

        if None == time:
            return self.interior_points
        else:
            return self.interior_points[np.isclose(self.interior_points[:, 0],
                                                   time)]

    def get_boundary_points(self, time=None):
        if not self.discreted:
            raise ValueError("Try get points on a not dicreted domain.")
        if time is not None and not self.time_dependent:
            raise ValueError(
                "Try get points at specific time on a time independent geometry domain."
            )

        if None == time:
            return self.boundary_points
        else:
            return self.boundary_points[np.isclose(self.boundary_points[:, 0],
                                                   time)]

    def get_initial_points(self):
        if not self.discreted or not self.time_dependent:
            raise ValueError(
                "Try get points on a not dicreted or a time independent domain."
            )

        time = np.sort(self.timedomain.get_points(), axis=None)[0]
        if self.timedomain.is_on_initial(time):
            return self.get_points(time)
        else:
            raise ValueError("The earliest time is not on initial")

    def get_points(self, time=None):
        if not self.discreted:
            raise ValueError("Try get points on a not dicreted domain.")
        if time is not None and not self.time_dependent:
            raise ValueError(
                "Try get points at specific time on a time independent geometry domain."
            )

        if None == time:
            return self.points
        else:
            return self.points[np.isclose(self.points[:, 0], time)]

    def get_npoints(self):
        if not self.discreted:
            raise ValueError(
                "This interface is only valid within dicreted domain.")
        return self.points.size // (
            self.dim +
            1) if self.time_dependent else self.points.size // self.dim


class Rectangle(Domain):

    def __init__(self, origins, extents):
        super(Rectangle, self).__init__(2, False, False)
        self.origins = np.array(origins, dtype=self.dtype)
        self.extents = np.array(extents, dtype=self.dtype)

        self.perimeter = 2 * np.sum(self.extents, dtype=self.dtype)
        self.volume = np.prod(self.extents, dtype=self.dtype)

    def _not_close_corners(self, points):
        l1 = self.extents[0]
        l2 = l1 + self.extents[1]
        l3 = l2 + l1

        points = points[np.logical_not(np.isclose(points,
                                                  l1 / self.perimeter))]
        points = points[np.logical_not(np.isclose(points,
                                                  l2 / self.perimeter))]
        points = points[np.logical_not(np.isclose(points,
                                                  l3 / self.perimeter))]

        return points

    def _sample_interior_points(self, n, sampler):
        x = sample(n, self.dim, sampler, self.dtype)
        return self.origins + self.extents * x

    def _sample_boundary_points(self, n, sampler):
        u = np.ravel(sample(n + 4, 1, sampler, self.dtype))
        u = self._not_close_corners(u) * self.perimeter
        points_list = [np.zeros((0, 2), dtype=self.dtype)]
        for l in u:
            if l < self.extents[0]:
                points_list.append((self.origins + [l, 0])[np.newaxis, :])
            elif l < np.sum(self.extents):
                points_list.append(
                    (self.origins +
                     [self.extents[0], l - self.extents[0]])[np.newaxis, :])
            elif l < np.sum(self.extents) + self.extents[0]:
                points_list.append((self.origins + [
                    self.perimeter / 2 + self.extents[0] - l, self.extents[1]
                ])[np.newaxis, :])
            else:
                points_list.append(
                    (self.origins + [0, self.perimeter - l])[np.newaxis, :])
        ret = np.concatenate(points_list, axis=0).astype(self.dtype)
        return ret

    def discrete(self, interior, boundary, sampler='uniform'):
        # Step1: some checks.
        if self.discreted:
            raise ValueError(
                "Do not support discrete a domian more than once.")

        if self.time_dependent:
            raise ValueError(
                "A domain should not be time dependent before discrete process."
            )

        if interior < 0 or boundary < 0:
            raise ValueError(
                "Invalid arguments, should sample at least 0 points from interior and boundary."
            )

        # Step2: discrete process
        self.interior_points = self._sample_interior_points(
            interior, sampler) if interior != 0 else np.zeros(
                (0, self.dim), self.dtype)
        if self.interior_points.shape[0] != interior:
            warnings.warn(
                "Expected {x} interior points, but get {y} interior points".
                format(x=interior, y=self.interior_points.shape[0]))
        self.boundary_points = self._sample_boundary_points(
            boundary, sampler) if boundary != 0 else np.zeros(
                (0, self.dim), self.dtype)
        if self.boundary_points.shape[0] != boundary:
            warnings.warn(
                "Expected {x} boundary points, but get {y} boundary points".
                format(x=boundary, y=self.boundary_points.shape[0]))

        self.points = np.concatenate(
            (self.interior_points, self.boundary_points), axis=0)

        self.discreted = True
        return self


# Helper class to compute Jacobian matrix for tensor with shape x[bsize, m] and y[bsize, n]
class Jacobian(object):

    def __init__(self, x, y):
        super(Jacobian, self).__init__()
        # Ensure that the shapes of x and y match
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise ValueError(
                'Invalid shape, the number of dimensions of x and y should be 2.'
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Invalid shape, the first dimension of x and y should be the same.'
            )
        self.bsize = x.shape[0]
        self.x = x
        self.y = y
        self.J = [None for _ in range(self.y.nelement() // self.bsize)]

    def _check_items(self, items):
        if isinstance(items, int):
            return True
        elif isinstance(items, tuple) or isinstance(items, list):
            if len(items) != 2:
                return False
            for item in items:
                if not isinstance(item, int):
                    return False
        else:
            return False
        return True

    def __getitem__(self, items):
        if not self._check_items(items):
            raise TypeError(
                'Invalid item, should be int or tuple/list of length 2.')

        # Compute the gradient only if it hasn't been computed yet
        i = items if isinstance(items, int) else items[0]
        if self.J[i] is None:
            self.J[i], = flow.autograd.grad(self.y[:, i],
                                            self.x,
                                            flow.ones_like(self.y[:, i]),
                                            create_graph=True,
                                            retain_graph=True)
        return self.J[items] if isinstance(
            items, int) else self.J[items[0]][:, items[1]]


class PDEs(object):

    def __init__(self, num_pdes, variables=None):
        self.num_pdes = num_pdes
        self.variables = variables

    def get_num_pdes(self):
        return self.num_pdes

    def get_variables(self):
        return self.variables

    def evaluate(self, inputs, outputs, inputs_names=None, outputs_names=None):
        raise NotImplementedError("Implement in PDEs subclass")


# Time independent
# du_dx + dv_dy = 0
# u * du_dx + v * du_dy - nu / rho * du_dx2 - nu / rho * du_dy2 + 1.0 / rho * dp_dx = 0
# u * dv_dx + v * dv_dy - nu / rho * dv_dx2 - nu / rho * dv_dy2 + 1.0 / rho * dp_dy = 0
class NavierStokes2D(PDEs):

    def __init__(self, nu=0.01, rho=1.0):
        super(NavierStokes2D, self).__init__(2)
        self.nu = nu
        self.rho = rho

    def evaluate(self,
                 inputs,
                 outputs,
                 inputs_names=['x', 'y'],
                 outputs_names=['u', 'v', 'p']):

        pos_x = inputs_names.index('x')
        pos_y = inputs_names.index('y')
        pos_u = outputs_names.index('u')
        pos_v = outputs_names.index('v')
        pos_p = outputs_names.index('p')
        u = outputs[:, pos_u]
        v = outputs[:, pos_v]
        jac = Jacobian(inputs, outputs)
        hes_u = Jacobian(inputs, jac[pos_u])
        hes_v = Jacobian(inputs, jac[pos_v])

        pde_1 = jac[pos_u, pos_x] + jac[pos_v, pos_y]
        pde_2 = u * jac[pos_u, pos_x] + v * jac[pos_u, pos_y] - self.nu / self.rho * hes_u[pos_x, pos_x] - \
                self.nu / self.rho * hes_u[pos_y, pos_y] + 1.0 / self.rho * jac[pos_p, pos_x]
        pde_3 = u * jac[pos_v, pos_x] + v * jac[pos_v, pos_y] - self.nu / self.rho * hes_v[pos_x, pos_x] - \
                self.nu / self.rho * hes_v[pos_y, pos_y] + 1.0 / self.rho * jac[pos_p, pos_y]

        return [pde_1, pde_2, pde_3]


class BC(object):

    def __init__(self, domain, constrain_func, value_func, boundary_points):
        if not domain.is_discreted():
            raise ValueError("The geometry must be discreted")
        self.domain = domain
        self.value_func = cache_func(value_func)
        self.constrain_func = constrain_func
        self.boundary_points = boundary_points

    def evaluate(self, net, value_indexes):
        raise NotImplementedError(
            "Boundary condition evaluate not implement in {}".format(
                self.__class__.__name__))


@cache_func
def get_boundary_constrain_points(domain,
                                  X,
                                  constrain_func,
                                  with_boundary_normal=False):

    if X is None:
        boundary_points = domain.get_boundary_points()
        constrain = constrain_func(boundary_points)
        if with_boundary_normal:
            boundary_points = boundary_points[constrain]
            n = tensor(domain.boundary_normal(boundary_points))
            boundary_points = tensor(boundary_points)
            boundary_points.requires_grad = True
            return boundary_points, n
        else:
            return tensor(boundary_points[constrain])
    else:
        constrain = domain.is_on_boundary(X) * constrain_func(X)
        if with_boundary_normal:
            boundary_points = X[constrain]
            n = tensor(domain.boundary_normal(boundary_points))
            boundary_points = tensor(boundary_points)
            boundary_points.requires_grad = True
            return boundary_points, n
        else:
            return tensor(X[constrain])


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self,
                 domain,
                 constrain_func,
                 value_func,
                 boundary_points=None):
        super().__init__(domain, constrain_func, value_func, boundary_points)

    def evaluate(self, net, value_indexes):
        if not check_indexes(value_indexes):
            raise ValueError("value indexes should be list/tuple of int ")

        boundary_points = get_boundary_constrain_points(
            self.domain, self.boundary_points, self.constrain_func, False)

        net_out = net(boundary_points)
        gt = self.value_func(boundary_points)
        diff = net_out[:, value_indexes] - gt
        return diff


class AISolver(object):

    def __init__(self, algorithm, network, loss, optimizer, checkpoint_path):
        self.algorithm = algorithm
        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def save_checkpoint(self, path):
        save_checkpoint(path, self.network, self.optimizer)

    def load_checkpoint(self, path):
        load_checkpoint(path, self.network, self.optimizer)

    def train(self):
        raise NotImplementedError("Implement in AISolver subclass")

    def predict(self):
        raise NotImplementedError("Implement in AISolver subclass")

    def visualize(self):
        raise NotImplementedError("Implement in AISolver subclass")


class PINNSolver(AISolver):

    def __init__(self,
                 network,
                 loss,
                 optimizer,
                 domain,
                 pdes,
                 inputs_names,
                 outputs_names,
                 bcs=[],
                 bc_indexes=[],
                 ics=[],
                 ic_indexes=[],
                 sups=[],
                 checkpoint_path='./log'):
        super(PINNSolver, self).__init__('PINNs', network, loss, optimizer,
                                         checkpoint_path)
        self.domain = domain
        self.pdes = pdes
        self.inputs_names = inputs_names
        self.outputs_names = outputs_names
        self.bcs = bcs
        self.bc_indexes = bc_indexes
        self.ics = ics
        self.ic_indexes = ic_indexes
        self.sups = sups

    def save_checkpoint(self, path):
        save_checkpoint(path, self.network, self.optimizer,
                        self.pdes.get_variables())

    def load_checkpoint(self, path):
        load_checkpoint(path, self.network, self.optimizer,
                        self.pdes.get_variables())

    def train(self, num_epoch, log_frequency=100, checkpoint_frequency=1000):
        for idx in range(num_epoch):
            self.optimizer.zero_grad()
            # PDE items
            inputs_interior = tensor(self.domain.get_interior_points())
            inputs_interior.requires_grad = True
            outputs_interior = self.network(inputs_interior)
            pde_items = self.pdes.evaluate(inputs_interior, outputs_interior,
                                           self.inputs_names,
                                           self.outputs_names)

            # bc items
            bc_items = [
                bc.evaluate(self.network, indexes)
                for bc, indexes in zip(self.bcs, self.bc_indexes)
            ]

            # ic items
            ic_items = [
                ic.evaluate(self.network, indexes)
                for ic, indexes in zip(self.ics, self.ic_indexes)
            ]

            # sup items
            sup_items = [
                self.network(sup['x']) - sup['y'] for sup in self.sups
            ]

            losses, loss = self.loss.evaluate(pde_items + bc_items + ic_items +
                                              sup_items)
            loss.backward()
            self.optimizer.step()

            if (idx + 1) % log_frequency == 0:
                print(
                    f"num_epoch: {idx + 1}, loss: {loss.detach().cpu().numpy():.7g}"
                )
                print(f"sub losses:")
                for loss_el in losses:
                    print(f"{loss_el.detach().cpu().numpy():.7g}")

                variables = self.pdes.get_variables()
                if variables is not None:
                    print(f"variables:")
                    for name in variables.names:
                        print(
                            f"{name}: {getattr(variables, name).detach().cpu().numpy():.7g}"
                        )
            if (idx + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(self.checkpoint_path + '/checkpoint_' +
                                     str(idx + 1) + '.pt')

    def evaluate(self, domain=None):
        domain = self.domain if domain is None else domain
        inputs = tensor(domain.get_points())
        return self.network(inputs).detach().cpu().numpy()

    def visualize(self, filename='outputs', domain=None, time=None):
        domain = self.domain if domain is None else domain
        inputs = tensor(domain.get_points())
        outputs = self.network(inputs).detach().cpu().numpy()
        if outputs.shape[1] == 1:
            save_vtk(filename, domain, outputs[:, 0], time)
        else:
            for idx in range(outputs.shape[1]):
                save_vtk(filename + '_' + str(idx), domain, outputs[:, idx],
                         time)
