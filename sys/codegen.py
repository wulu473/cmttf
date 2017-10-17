#!/usr/bin/env python3
"""
Script to generate jacobian from flux function
"""
import argparse
import logging
import sys
import math
import sympy as sp
import numpy as np

class System:
    """
        Base class for defining a system of governing equations
    """
    x, t = sp.symbols("x t")
    dx, x_i = sp.symbols("dx x_i")

    def diff_f(self, f, x, o=1, dx=sp.symbols("dx")):
        """
            Compute a finite difference approximation of f w.r.t x
            In:
                - f: Function of which to take the derivative
                - x: Independent variable
                - o: Order of derivtive
                - dx: Grid spacing
            Out:
                - finite difference
        """
        ss = int(math.ceil(o/2.)) # stencil size
        x_is = [ x + i*dx for i in range(-ss,ss+1) ]
        return sp.differentiate_finite(f,x,o,points=x_is)


    def __init__(self):
        logging.info("Initialising system...")
        self.F_disc = None
        self.F_disc_lin = None
        self.constants = {}
        self.define()

        if self.F_disc == None:
            self.discretise()
        self.F_disc = sp.Matrix(self.F_disc)
        self.F_disc = self.F_disc.expand()
        self.F_disc_lin = sp.Matrix(self.F_disc_lin)
        self.F_disc_lin = self.F_disc_lin.expand()
        self.stencilSize = self._stencil_size()
        self.stateSize = len(self.state)


    def discretise(self):
        # TODO Implement a method that automatically deduces F_disc from F
        raise NotImplementedError("Implement a discretisce method that deduces F_disc from F")

    def replace_symbolic_constants(self):
        return self.F_disc.subs(self.constants)

    def _stencil_size(self):
        self.stencilSize = 0
        for u in self.state:
            # Assume stencil isn't larger than 20
            # TODO make this more elegant -- a lot more elegant
            for di in range(20):
                for f_i in self.F_disc:
                    if f_i.coeff(u(self.x_i+di*self.dx)) != 0:
                        self.stencilSize=max(self.stencilSize,di)
        return self.stencilSize


class Schwartz(System):
    """
        System used in
        Moriarty, J. A., Schwartz, L. W., & Tuck, E. O. (1991).
        Unsteady spreading of thin liquid films with small surface tension.
        Physics Of Fluids A Fluid Dynamics, 3(5), 733. http://doi.org/10.1063/1.858006
    """
    def set_parameters(self, epsilon):
        self.constants = {
            self.epsilon : epsilon
        }

    def define(self):
        self.epsilon = sp.symbols("epsilon")
        self.u = sp.symbols("u")
        self.state = [self.u]

        # We solve \partial_t u = - F
        diff = sp.diff
        self.F = [diff( self.u(self.x)**3 + self.epsilon**3*self.u(self.x)**3*diff(self.u(self.x),self.x,3),self.x)]

        diff = self.finite_diff
        self.F_disc = [diff( self.u(self.x)**3 + self.epsilon**3*self.u(self.x)**3*diff(self.u(self.x),self.x,3),self.x).subs(self.x,self.x_i)]


class Myers(System):
    def define(self):
        self.mu, self.sigma, self.tau, self.beta = sp.symbols("mu sigma tau beta")
        # g1 is parallel to substrate, g2 is perp both are multiplied by rho_w as in Myers 2002
        self.g1, self.g2 = sp.symbols("g1 g2")
        self.h = sp.symbols("h")
        self.h0= sp.symbols("h0")
        self.state = [self.h]

        mu = self.mu
        dx = self.dx
        sigma = self.sigma
        (g1, g2) = (self.g1, self.g2)

        # We solve \partial_t u = - F
        hLL = self.h(self.x_i-self.dx*2)
        hL = self.h(self.x_i-self.dx)
        hC = self.h(self.x_i)
        hR = self.h(self.x_i+self.dx)
        hRR = self.h(self.x_i+self.dx*2)

        hLC = (hL+hC)/2
        hCR = (hC+hR)/2

        h0L = self.h0(self.x_i-self.dx)
        h0C = self.h0(self.x_i)
        h0R = self.h0(self.x_i+self.dx)

        h0LC = (h0L+h0C)/2
        h0CR = (h0C+h0R)/2

        stress = self.tau*(hCR**2-hLC**2)/(2*mu*dx)
        tension = hCR**3/(3*mu*dx**4)*sigma*(-hL+3*hC-3*hR+hRR) - \
                  hLC**3/(3*mu*dx**4)*sigma*(-hLL+3*hL-3*hC+hR)
        gravity = hCR**3/(3*mu*dx**2)*g2*(hR-hC) - \
                  hLC**3/(3*mu*dx**2)*g2*(hC-hL) - \
                  (hCR**3/(3*mu*dx)*g1 - hLC**3/(3*mu*dx)*g1)

        self.F_disc = [(stress + tension + gravity - self.beta).subs(self.x, self.x_i)]

        stress_lin = self.tau*(h0CR**2-h0LC**2)/(2*mu*dx)
        tension_lin = h0CR**3/(3*mu*dx**4)*sigma*(-hL+3*hC-3*hR+hRR) - \
                      h0LC**3/(3*mu*dx**4)*sigma*(-hLL+3*hL-3*hC+hR)
        gravity_lin = h0CR**3/(3*mu*dx**2)*g2*(hR-hC) - \
                      h0LC**3/(3*mu*dx**2)*g2*(hC-hL) - \
                      (h0CR**3/(3*mu*dx)*g1 - h0LC**3/(3*mu*dx)*g1)
        # beta is already linear
        self.F_disc_lin = [(stress_lin + tension_lin + gravity_lin - self.beta).subs(self.x, self.x_i)]

    def set_parameters(self, mu, sigma, g1, g2, tau, beta):
        self.constants = {
            self.mu : mu,
            self.sigma : sigma,
            self.tau: tau,
            self.g1 : g1,
            self.g2 : g2,
            self.beta: beta
        }

        
 
class Roberts1998(System):
    def set_parameters(self,g,g1,g2,mu,tau,sigma,rho,beta):
        self.constants = {
            self.g     : g,
            self.g1    : g1,
            self.g2    : g2,
            self.mu    : mu, 
            self.tau   : tau,
            self.sigma : sigma,
            self.rho   : rho,
            self.beta  : beta
        }

    def define(self):
        self.g,self.g1,self.g2,self.mu,self.tau= sp.symbols("g g1 g2 mu tau")
        self.sigma, self.rho, self.beta = sp.symbols("sigma rho beta")

        self.u, self.eta = sp.symbols("eta u")
        self.state = [self.eta, self.u]

        eta = self.eta
        u = self.u
        x = self.x
        x_i = self.x_i
        g1 = self.g1
        g2 = self.g2
        g = self.g
        mu = self.mu
        tau = self.tau
        sigma = self.sigma
        rho = self.rho
        beta = self.beta

        diff = self.diff_f
        # The coefficient for the shear stress term makes sense
        # when we consider that the coefficients in Myers model
        # differ by a factor of 2/3
        self.F_disc = [ diff(eta(x)*u(x),x) - beta,
                  +mu/rho*(2.467 - 0.3*diff(eta(x),x)**2)*u(x)/eta(x)**2 \
                  -0.82245*sigma/rho*diff(eta(x),x,3) \
                  -0.82245*g*(g1 + g2*diff(eta(x),x)) \
                  -1.234*tau/rho/eta(x) \
                  -4.093*mu/rho/eta(x)**1.323*diff(eta(x)**1.466*diff(u(x)/eta(x)**0.143,x),x) \
                  +1.5041*u(x)/eta(x)**0.0985*diff(eta(x)**0.0985*u(x),x)]
        self.F_disc = sp.Matrix(self.F_disc)
        self.F_disc = self.F_disc.subs(x,x_i)

    def reflect(self,state):
        state[1] *= -1.0
        return state

class Roberts1998Reduce(System):
    h, u, k, x = sp.symbols("h u k x")
    re, tau, b, g1, g2 = sp.symbols("Re tau b g1 g2")
    dx, x_i = sp.symbols("dx x_i")

    def set_parameters(self,re,g1,g2,b,tau):
        self.constants = {
                self.g1: g1*b,
                self.g2: g2*b,
                self.re: re,
                self.tau: tau
        }


    def define(self):
        self.state = [self.h,
                      self.u]

        self._set_F_disc_from_file()
        self.F_disc = self.F_disc.subs(self.x,self.x_i)

    def _set_F_disc_from_file(self):
        gamma,w,r = sp.symbols("gam w r")
        d = sp.symbols("d")
        replacements  = [ (self.h(i), self.diff(self.h(self.x),self.x,i)) for i in range(0,5)]
        replacements += [ (w(i), self.diff(self.u(self.x),self.x,i)) for i in range(0,5)]
        replacements += [ (self.k(i), 0) for i in range(0,5)] # Flat substrate
        replacements += [ (r,self.re) ]

        eqs = []
        with open("/home/raid/lw473/code/reduce/roberts-1998-4-6.out") as f:
            for line in f:
                line = line.strip().rstrip().split("%")[0]
                if not line: continue
                expr=line.split("=")[1]

                expr = sp.sympify(expr)
                expr = expr.subs(d,1)
                expr = expr.subs(gamma,1)
                expr = expr.subs(replacements)
                eqs.append(expr)

        self.F_disc = -sp.Matrix(eqs)

def main():
    """
        main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', help='Name of the model')
    args = parser.parse_args()

    System = getattr(sys.modules[__name__],args.MODEL)
    system = System()

    # Set SystemAttributes file
    print("stencilSize =", str(system.stencilSize))
    print("stateSize =", str(system.stateSize))


    # F_disc comes in a form with its variables of the form 'u(x_i+dx)'
    # This is not a form we can convert to C-code. Therefore introduce temporary variables
    # and define them in the C-code
    x_i, dx = sp.symbols("x_i dx")
    discretised_variables = []
    discretised_variables0 = []
    for i, stencil_i in enumerate(range(-system.stencilSize, 1+system.stencilSize)):
        for j, state_j in enumerate(system.state):
            state_tuple = (state_j(x_i+stencil_i*dx), sp.sympify("u_%i_%i"%(i, j)))
            state_tuple0 = (sp.sympify("%s0"%(state_j))(x_i+stencil_i*dx), sp.sympify("u0_%i_%i"%(i, j)))
            discretised_variables.append(state_tuple)
            discretised_variables0.append(state_tuple0)

    function = system.F_disc.subs(discretised_variables)
    for i, f_i in enumerate(function):
        print("F[%i] = %s;"%(i, sp.ccode(f_i)))

    function_lin = system.F_disc_lin.subs(discretised_variables0)
    function_lin = function_lin.subs(discretised_variables)
    for i, f_lin_i in enumerate(function_lin):
        print("F_lin[%i] = %s;"%(i, sp.ccode(f_lin_i)))

    for j, (_, u_j) in enumerate(discretised_variables):
        for i, f_i in enumerate(function):
            print("J(%i,%i) = %s;"%(i, j, sp.ccode(f_i.diff(u_j))))

    for j, (_, u_j) in enumerate(discretised_variables):
        for i, f_lin_i in enumerate(function_lin):
            print("J_lin(%i,%i) = %s;"%(i, j, sp.ccode(f_lin_i.diff(u_j))))

if __name__ == "__main__":
    main()
