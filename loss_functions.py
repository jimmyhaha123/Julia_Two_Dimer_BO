import subprocess

def single_dimer_ngspice_loss(p):
    gainr1, gainr2, resc1, resc2, lam, factor = p
    input_args = ['ngspice'] + [str(gainr1), str(gainr2), str(resc1), str(resc2), str(lam), str(factor)]
    result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
    result = float(result.decode("utf-8").strip())
    return result

def single_dimer_cmt_loss(p):
    w2, k, n11, n10, n20 = p
    input_args = ['cmt'] + [str(w2), str(k), str(n11), str(n10), str(n20), '-1000.0']
    result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
    result = float(result.decode("utf-8").strip())
    return result