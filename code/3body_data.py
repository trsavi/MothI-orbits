# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 03:57:55 2022

@author: Korisnik
"""




import pandas as pd

path = r'C:\Users\Korisnik\OneDrive\Desktop\3-body\DATA.dat'




with open(path) as f:
    lines = f.readlines() # Data starts at 7th line

# =============================================================================
# The initial conditions and periods T of periodic orbits for the non-hierarchical three-body problem in the case of  
# r_1(0) = (x_1, 0), r_2(0) = (1, 0), r_3(0) = (0, 0), r_1'(0) = (0, v_1), r_2'(0) = (0, v_2),  
# r_3'(0) = (0, -(m_1*v_1 + m_2*v2)/m_3) with the gravitational constant G = 1. The stability of periodic orbits 
# can be linear stable (S) or linear unstable (U).
# =============================================================================
    
def v3(m1, m2, v1, v2, m3 = 1): # y component of velocity of third body
    return(-(m1*v1+m2*v2)/m3)

def E(m1, m2, x1, v1, v2, m3 = 1, x2 = 1, x3 = 0): # Energy of the system
    e = (m1*v1**2 + m2*v2**2 + m3*(v3(m1,m2,v1,v2))**2)/2 - m1*m2/abs(x1-x2) - m1*m3/abs(x1-x3) - m2*m3/abs(x2-x3)
    return(e)

def L(m1, m2, x1, v1, v2, m3 = 1, x2 = 1, x3 = 0): # Angular momenutm of the system
    l = m1*x1*v1 + m2*x2*v2 + m3*x3*v3(m1,m2,v1,v2)
    return(l)

def T_E(t, e): # Scale-invariant period
    return(t*abs(e)**(1.5))


def L_E(l, e): # Scale-invariant angular momentum
    return(l*abs(e)**(0.5))


clmns = [x for x in lines[5].split() if x != '|'] + ['E', 'L', 'T_E', 'L_E', 'M_tot']

data = [line.split() for line in lines[7:]]


for line in data:
    m1, m2, x1, v1, v2, t = float(line[0]), float(line[1]), float(line[3]), float(line[4]), float(line[5]), float(line[6])
    e = E(m1, m2, x1, v1, v2)
    l = L(m1, m2, x1, v1, v2)
    t_e = T_E(t, e)
    l_e = L_E(l, e)
    M_tot = m1 + m2 + 1
    line += [e, l, t_e, l_e, M_tot]


df = pd.DataFrame(data, columns = clmns)

df.to_csv(r'C:\Users\Korisnik\OneDrive\Desktop\3-body\NEW_DATA.csv')












