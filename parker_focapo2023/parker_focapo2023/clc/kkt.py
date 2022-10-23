from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix


def get_equality_constrained_kkt_matrix(nlp, exclude_hessian=False):
    # To extend this to bound/inequality constrained problems, we would
    # need to add the Z/X term to the Hessian block
    kkt = BlockMatrix(2, 2)
    eq_jac = nlp.evaluate_jacobian_eq()
    if not exclude_hessian:
        hess = nlp.evaluate_hessian_lag()
        kkt.set_block(0, 0, hess)
    kkt.set_block(1, 0, eq_jac)
    kkt.set_block(0, 1, eq_jac.transpose())
    return kkt
