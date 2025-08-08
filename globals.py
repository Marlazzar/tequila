from sympy import jacobi_normalized

from tequila import grouping


def init():
    global calls
    calls = 0

    global jobid 
    jobid = None

    global groupings 
    groupings = None

    global counts 
    counts = None