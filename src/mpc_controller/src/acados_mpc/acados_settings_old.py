from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver

from mpc_controller.acados_mpc.bicycle_model import bicycle_model_old

import scipy.linalg
import numpy as np
import math

DEG2RAD = math.pi/180.0
RAD2DEG = 180.0/math.pi


def acados_settings(Tf, N):
    # create render arguments
    ocp = AcadosOcp()
    dt = Tf/N
    # export model
    model, constraint = bicycle_model(dt)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # define constraint
    model_ac.con_h_expr = constraint.expr

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    nsbx = 0
    nh = constraint.expr.shape[0]
    nsh = nh - 9 # # not slacking obstacle avoidance constraints
    # nsh = nh - 10 # # not slacking obstacle avoidance constraints & border

    ns = nsh + nsbx

    # discretization
    ocp.dims.N = N
    
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    unscale = N / Tf
    # unscale = 1.0

    # # car obstacle
    # # set cost
    # Q = np.diag([ 
    #     0, # s 
    #     0, # n
    #     5e-4, # n_diff
    #     0, # alpha
    #     0, # v
    #     1e-4, # v_diff
    #     0, # D
    #     0, # delta
    #     0, # yaw_rate

    # ])
    
    # R = np.eye(nu)
    # R[0, 0] = 1e-2
    # R[1, 1] = 1e-1
    
    # # set terminal cost
    # Qe = np.diag([ 
    #     0, # s
    #     0, # n
    #     5e-4, # n_diff
    #     0, # alpha
    #     0, # v
    #     1e-4, # v_diff
    #     0, # D
    #     0, # delta
    #     0, # yaw_rate
    # ])
    # car obstacle

    """ obstacle avoidance v2"""
    # Q = np.diag([ 
    #     1e-8, # s 
    #     1e-9, # n
    #     0, # n_diff
    #     1e-0, # alpha
    #     0, # v
    #     0, # v_diff
    #     0, # D
    #     0, # delta
    #     # 0, # yaw_rate

    # ])
    
    # R = np.eye(nu)
    # R[0, 0] = 1e0
    # R[1, 1] = 1e1
    
    # # set terminal cost
    # Qe = np.diag([ 
    #     1e-2, # s
    #     1e-5, # n
    #     0, # n_diff
    #     1e-0, # alpha
    #     0, # v
    #     0, # v_diff
    #     0, # D
    #     0, # delta
    #     # 0, # yaw_rate
    # ])
    """ ------------------ """
    """ V3 """
    # # follows track
    # OBS WITH S CONTROL
    Q = np.diag([ 
        0, # s 
        0, # n
        1e-8, # n_diff
        1e-8, # alpha
        0, # v
        1e0, # v_diff
        0, # D
        0, # delta
        # 0, # yaw_rate

    ])
    
    R = np.eye(nu)
    R[0, 0] = 1e2
    R[1, 1] = 8e1
    
    # set terminal cost
    Qe = np.diag([ 
        0, # s
        0, # n
        0, # n_diff
        1e-8, # alpha
        0, # v
        0, # v_diff
        0, # D
        0, # delta
        # 0, # yaw_rate
    ])
    """ ------------------ """
    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[8, 0] = 1.0
    Vu[9, 1] = 1.0
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # set intial references
    ocp.cost.yref = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0, 0, 0,])

    ocp.constraints.lbu = np.array([model.dthrottle_min, model.ddelta_min])
    ocp.constraints.ubu = np.array([model.dthrottle_max, model.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.parameter_values = np.array([0, 0]*6)
    ocp.constraints.lh = np.array(
        [
            constraint.along_min,
            constraint.alat_min,
            model.n_min,
            model.v_min,
            model.throttle_min,
            model.delta_min,
            constraint.dist_obs1_min,
            constraint.dist_obs2_min,
            constraint.dist_obs3_min,
            constraint.dist_obs4_min,
            constraint.dist_obs5_min,
            constraint.dist_obs6_min
        ]
    )
    ocp.constraints.uh = np.array(
        [
            constraint.along_max,
            constraint.alat_max,
            model.n_max,
            model.v_max,
            model.throttle_max,
            model.delta_max,
            constraint.dist_obs1_max,
            constraint.dist_obs2_max,
            constraint.dist_obs3_max,
            constraint.dist_obs4_max,
            constraint.dist_obs5_max,
            constraint.dist_obs6_max
        ]
    )

    slack_L1_cost = np.array([
        1e1,
        1e1,
        1e3, ##
        # 1,
        # 1,
    ])
    slack_L2_cost = np.array([
        1e3,
        1e3,
        1e3, ##
        # 1,
        # 1,
    ])

    ocp.cost.zl = slack_L1_cost
    ocp.cost.zu = slack_L1_cost
    ocp.cost.Zl = slack_L2_cost
    ocp.cost.Zu = slack_L2_cost



    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    # ocp.constraints.idxsh = np.array(range(nsh)) # = [0,1,2]
    # ocp.constraints.idxsh = np.array([0,1,2,4,5]) # = [0,1,2]
    ocp.constraints.idxsh = np.array([0,1,2]) # = [0,1,2]

    # set intial condition
    ocp.constraints.x0 = model.x0

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 4
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.tol = 1e-3
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    # acados_solver = AcadosOcpSolver.generate(ocp, json_file="acados_ocp.json")

    acados_integrator = AcadosSimSolver(ocp, json_file="acados_ocp.json")
    # acados_solver = AcadosOcpSolver.

    return constraint, model, acados_solver, acados_integrator