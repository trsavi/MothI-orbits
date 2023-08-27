# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:35:13 2022

@author: Korisnik
"""




#%% LIBRARIES AND READING THE FILE

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math as math
from sklearn.metrics import r2_score




path = r'C:\Users\Korisnik\OneDrive\Desktop\3-body\CHINA_CSV_3_BODY_DATA.csv'

df = pd.read_csv(path)






#%%
linreg = LinearRegression().fit(df[['m_1', 'm_2']], df['T_E'])

a1, a2, b = linreg.coef_[0], linreg.coef_[1], linreg.intercept_ # 24.55001976 16.54889656] -16.877929197261594

T_E_predicted = linreg.predict(df[['m_1', 'm_2']])

r2 = r2_score(df['T_E'], T_E_predicted)
print(r2) # 0.9958678037142547

T_E_errors = df['T_E'] - T_E_predicted

plt.scatter(df['T_E'], T_E_errors)
plt.show()
    

#%%
fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

ax.plot(df['m_1'], df['m_2'], df['T_E'])
plt.show()

#%%
plt.tricontourf(df['m_1'], df['m_2'], df['T_E'], levels = 11)
plt.colorbar()
plt.show()

#%% ABSOLUTE VALUE OF L^* ON M1-M2 PLANNE
plt.tricontourf(df['m_1'], df['m_2'], abs(df['L_E']), levels = 11)
cbar = plt.colorbar()
cbar.set_label('$|L^*|$', rotation = 0, fontsize = 14, labelpad=17)


plt.xlabel(r'm$_1$', fontsize = 14)
plt.ylabel(r'm$_2$', fontsize = 14)

plt.show()

#%% STABLE AND UNSTABLE DATA

df_stable = df[df['stability'] == 'S'] # Data for stable orbits
df_unstable = df[df['stability'] == 'U'] # Data for unstable orbits


#%% STABLE AND UNSTABLE DATA FOR M1 >= M2 AND M1 < M2

df_1 = df[df['m_1'] >= df['m_2']] # Data for M1 >= M2
df_stable_1 = df_1[df_1['stability'] == 'S'] # Data for stable orbits
df_unstable_1 = df_1[df_1['stability'] == 'U'] # Data for unstable orbits


df_2 = df[df['m_1'] <= df['m_2']] # Data for M1 < M2 
df_stable_2 = df_2[df_2['stability'] == 'S'] # Data for stable orbits
df_unstable_2 = df_2[df_2['stability'] == 'U'] # Data for unstable orbits


#%% PLOTTING STABLE VS UNSTABLE ORBITS FOR M1 >= M2

plt.scatter(df_unstable_1['m_1'], df_unstable_1['m_2'], label = 'Unstable')
plt.scatter(df_stable_1['m_1'], df_stable_1['m_2'], label = 'Stable')

plt.xlabel('m$_1$', fontsize = 14)
plt.ylabel('m$_2$', fontsize = 14)
plt.legend(fontsize = 14)

plt.show()


#%% PLOTTING STABLE VS UNSTABLE ORBITS FOR M1 < M2

plt.scatter(df_unstable_2['m_1'], df_unstable_2['m_2'], label = 'Unstable')
plt.scatter(df_stable_2['m_1'], df_stable_2['m_2'], label = 'Stable')

plt.xlabel('m$_1$', fontsize = 14)
plt.ylabel('m$_2$', fontsize = 14)
plt.legend(fontsize = 14)

plt.show()


#%% PLOTTING STABLE VS UNSTABLE ORBITS

plt.scatter(df['m_1'], df['m_2'])
plt.scatter(df_stable['m_1'], df_stable['m_2'])
plt.show()




#%%
plt.tricontourf(df['m_1'], df['m_2'], df['T_E'], levels = 11)
plt.scatter(df_stable['m_1'], df_stable['m_2'], alpha = 0.1)
plt.colorbar()
plt.show()





#%% PLOTTING INVARIANT PERIOD VS INVARIANT ANGULAR MOMENTUM

plt.scatter(df_unstable['T_E'], df_unstable['L_E'])
plt.scatter(df_stable['T_E'], df_stable['L_E'])

plt.xlabel('T_E')
plt.ylabel('L_E')

plt.show()

#%% PLOTTING INVARIANT PERIOD VS INVARIANT ANGULAR MOMENTUM FOR M1 >= M2

plt.scatter(df_unstable_1['T_E'], df_unstable_1['L_E'], label = 'Unstable')
plt.scatter(df_stable_1['T_E'], df_stable_1['L_E'], label = 'Stable')

plt.xlabel('$T^*$', fontsize = 14)
plt.ylabel('$L^*$', fontsize = 14, labelpad=0)
plt.legend(fontsize = 14)

plt.show()

#%% PLOTTING INVARIANT PERIOD VS INVARIANT ANGULAR MOMENTUM FOR M1 < M2

plt.scatter(df_unstable_2['T_E'], df_unstable_2['L_E'], label = 'Unstable')
plt.scatter(df_stable_2['T_E'], df_stable_2['L_E'], label = 'Stable')

plt.xlabel('$T^*$', fontsize = 14)
plt.ylabel('$L^*$', fontsize = 14, labelpad=0)
plt.legend(fontsize = 14)

plt.show()


#%% PLOTTING INVARIANT PERIOD VS ABSOLUTE INVARIANT ANGULAR MOMENTUM

plt.scatter(df_unstable['T_E'], abs(df_unstable['L_E']))
plt.scatter(df_stable['T_E'], abs(df_stable['L_E']))

plt.xlabel('$T^*$', fontsize = 14)
plt.ylabel('$|L^*|$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()

#%% PLOTTING INVARIANT PERIOD VS ABSOLUTE INVARIANT ANGULAR MOMENTUM FOR M1 >= M2

plt.scatter(df_unstable_1['T_E'], abs(df_unstable_1['L_E']))
plt.scatter(df_stable_1['T_E'], abs(df_stable_1['L_E']))

plt.xlabel('$T^*$', fontsize = 14)
plt.ylabel('$|L^*|$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()


#%% PLOTTING INVARIANT PERIOD VS ABSOLUTE INVARIANT ANGULAR MOMENTUM FOR M1 < M2

plt.scatter(df_unstable_1['T_E'], abs(df_unstable_2['L_E']))
plt.scatter(df_stable_1['T_E'], abs(df_stable_2['L_E']))

plt.xlabel('$T^*$', fontsize = 14)
plt.ylabel('$|L^*|$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()



#%% PLOTTING TOTAL MASS TO THE POWER 5/2 VS INVARIANT PERIOD

plt.scatter(df_unstable['M_tot']**(5/2), df_unstable['T_E'])
plt.scatter(df_stable['M_tot']**(5/2), df_stable['T_E'])

plt.xlabel(r'$M_{tot}^{5/2}$', fontsize = 14, labelpad = -4)
plt.ylabel('$T^*$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.plot()


#%% PLOTTING TOTAL MASS TO THE POWER 5/2 VS INVARIANT PERIOD FOR M1 >= M2

plt.scatter(df_unstable_1['M_tot']**(5/2), df_unstable_1['T_E'])
plt.scatter(df_stable_1['M_tot']**(5/2), df_stable_1['T_E'])

plt.xlabel(r'$M_{tot}^{5/2}$', fontsize = 14, labelpad = -4)
plt.ylabel('$T^*$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.plot()


#%% PLOTTING TOTAL MASS TO THE POWER 5/2 VS INVARIANT PERIOD FOR M1 < M2

plt.scatter(df_unstable_2['M_tot']**(5/2), df_unstable_2['T_E'])
plt.scatter(df_stable_2['M_tot']**(5/2), df_stable_2['T_E'])

plt.xlabel(r'$M_{tot}^{5/2}$', fontsize = 14, labelpad = -4)
plt.ylabel('$T^*$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.plot()

#%% abs_L_E_vs_M_tot_13_6

plt.scatter(df_unstable['M_tot']**(13/6), abs(df_unstable['L_E']))
plt.scatter(df_stable['M_tot']**(13/6), abs(df_stable['L_E']))

plt.xlabel('$M_{tot}^{13/6}$', fontsize = 14, labelpad = 0)
plt.ylabel('$|L^*|$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()


#%% abs_L_E_vs_M_tot_13_6 for M1 >= M2

plt.scatter(df_unstable_1['M_tot']**(13/6), abs(df_unstable_1['L_E']))
plt.scatter(df_stable_1['M_tot']**(13/6), abs(df_stable_1['L_E']))

plt.xlabel('$M_{tot}^{13/6}$', fontsize = 14, labelpad = 0)
plt.ylabel('$|L^*|$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()


#%% abs_L_E_vs_M_tot_13_6 for M1 < M2

plt.scatter(df_unstable_2['M_tot']**(13/6), abs(df_unstable_2['L_E']))
plt.scatter(df_stable_2['M_tot']**(13/6), abs(df_stable_2['L_E']))

plt.xlabel('$M_{tot}^{13/6}$', fontsize = 14, labelpad = 0)
plt.ylabel('$|L^*|$', fontsize = 14)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()

#%% PLOTTING MASS INDEPENDENT INVARIANT PERIOD AND ANGULAR MOMENTUM FOR M1 >= M2

plt.scatter(df_unstable_1['T_E']/df_unstable_1['M_tot']**(5/2), abs(df_unstable_1['L_E'])/df_unstable_1['M_tot']**(13/6))
plt.scatter(df_stable_1['T_E']/df_stable_1['M_tot']**(5/2), abs(df_stable_1['L_E'])/df_stable_1['M_tot']**(13/6))

plt.xlabel('$T_{s.i.}$', fontsize = 14, labelpad = 0)
plt.ylabel('$|L_{s.i.}|$', fontsize = 14, labelpad = 0)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()


#%% PLOTTING MASS INDEPENDENT INVARIANT PERIOD AND ANGULAR MOMENTUM FOR M1 < M2

plt.scatter(df_unstable_2['T_E']/df_unstable_2['M_tot']**(5/2), abs(df_unstable_2['L_E'])/df_unstable_2['M_tot']**(13/6))
plt.scatter(df_stable_2['T_E']/df_stable_2['M_tot']**(5/2), abs(df_stable_2['L_E'])/df_stable_2['M_tot']**(13/6))

plt.xlabel('$T_{s.i.}$', fontsize = 14, labelpad = 0)
plt.ylabel('$|L_{s.i.}|$', fontsize = 14, labelpad = 0)
plt.legend(['Unstable', 'Stable'], fontsize = 14)

plt.show()


"""
#%% STABLE AND UNSTABLE MASSES


M_tot_unstable = np.array([m for m in df_unstable['M_tot']])
M_tot_stable = np.array([m for m in df_stable['M_tot']])


#%%
plt.scatter(M_tot_unstable**(5/2), df_unstable['T_E'])
plt.scatter(M_tot_stable**(5/2), df_stable['T_E'])

plt.xlabel('M_tot**(5/2)')
plt.ylabel('T_E')
plt.legend(['Unstable', 'Stable'])

plt.plot()

#%%

linreg = LinearRegression().fit((M_tot_stable**(5/2)).reshape(-1, 1), df_stable['T_E'])

a, b = linreg.coef_[0], linreg.intercept_

print(a, b) # 1.5513079546371664 -0.09742697643231324 for 5/2 and 19.495743920101525 -34.27967390545825 for 1

T_E_predicted = linreg.predict((M_tot_stable**(5/2)).reshape(-1, 1))


r2 = r2_score(df_stable['T_E'], T_E_predicted)
print(r2) # 0.9991584162465774 for M_tot**(5/2) and 0.9980303802605666 for M_tot

T_E_errors = df_stable['T_E'] - T_E_predicted

plt.scatter(df_stable['T_E'], T_E_errors)
plt.show()




#%%

plt.scatter(M_tot_unstable**(13/6), pos_L_E_unstable)
plt.scatter(M_tot_stable**(13/6), pos_L_E_stable)

plt.xlabel('M_tot**(13/6)')
plt.ylabel('|L_E|')
plt.legend(['Unstable', 'Stable'])

plt.show()



#%%

# =============================================================================
# linreg = LinearRegression().fit(M_tot_momentum_stable.reshape(-1, 1), df_stable['L_E'])
# 
# a, b = linreg.coef_[0], linreg.intercept_
# 
# print(a, b)
# 
# L_E_predicted = linreg.predict(M_tot_momentum_stable.reshape(-1, 1))
# 
# 
# r2 = r2_score(df_stable['T_E'], L_E_predicted)
# print(r2)
# =============================================================================
"""
#%%


pos_sym_L_E = np.array([])

Masses = np.array([m for m in zip(df['m_1'], df['m_2'], df['L_E'])])


mass_dict = {} # mass_1: [mass_2]

sym_mass_1 = np.array([])
sym_mass_2 = np.array([])

for mass in Masses:
    if mass[0] in mass_dict:
        mass_dict[mass[0]].append([mass[1], mass[2]])
    else:
        mass_dict[mass[0]] = [[mass[1], mass[2]]]
        
    if mass[1] in mass_dict:
        for m in mass_dict[mass[1]]:
            if m[0] == mass[0]:
                sym_mass_1 = np.append(sym_mass_1, [mass[0], mass[1]])
                sym_mass_2 = np.append(sym_mass_2, [mass[1], mass[0]])
                l = abs((mass[2] + m[1])/2)
                pos_sym_L_E = np.append(pos_sym_L_E, [l, l])
            
     
            
#%%

print(len(sym_mass_1), len(pos_sym_L_E))

plt.tricontourf(sym_mass_1, sym_mass_2, pos_sym_L_E, levels = 11)
plt.colorbar()
plt.axis('equal')
plt.title('Symmetric plot of angular momentum |L1+L2|/2')
plt.xlabel('m_sym')
plt.ylabel('m_sym')
plt.show()

            
            
            
            
            
























#%%

pct = 1.05
error = 0.0004

pdf_stable = df_stable_1[ df_stable_1['m_1'].between(pct*df_stable_1['m_2']-error, pct*df_stable_1['m_2']+error) ]
pdf_unstable = df_unstable_1[ df_unstable_1['m_1'].between(pct*df_unstable_1['m_2']-error, pct*df_unstable_1['m_2']+error) ]

plt.scatter( pdf_unstable['T_E']/pdf_unstable['M_tot']**(5/2), abs(pdf_unstable['L_E'])/pdf_unstable['M_tot']**(13/6), label = 'Unstable')
plt.scatter( pdf_stable['T_E']/pdf_stable['M_tot']**(5/2), abs(pdf_stable['L_E'])/pdf_stable['M_tot']**(13/6), label = 'Stable')


plt.xlabel('$T_{s.i.}$', fontsize = 14)
plt.ylabel('$|L_{s.i.}|$', fontsize = 14)

plt.legend(['Unstable', 'Stable'], fontsize = 14)
plt.tight_layout()


plt.show()



#%% PLOTTING SLICES IN M1 M2 SPACE OF T_S.I. VS |L_S.I.| FOR M1 BIGGER THAN M2


PCT = [1.00 + i*0.05 for i in range(5)]

for pct in PCT:
    
    pdf_stable = df_stable_1[ df_stable_1['m_1'].between(pct*df_stable_1['m_2']-0.001, pct*df_stable_1['m_2']+0.001) ]
    pdf_unstable = df_unstable_1[ df_unstable_1['m_1'].between(pct*df_unstable_1['m_2']-0.001, pct*df_unstable_1['m_2']+0.001) ]

    plt.scatter( pdf_unstable['T_E']/pdf_unstable['M_tot']**(5/2), abs(pdf_unstable['L_E'])/pdf_unstable['M_tot']**(13/6), label = 'Unstable')
    plt.scatter( pdf_stable['T_E']/pdf_stable['M_tot']**(5/2), abs(pdf_stable['L_E'])/pdf_stable['M_tot']**(13/6), label = 'Stable')


    plt.xlabel('$T_{s.i.}$', fontsize = 14)
    plt.ylabel('$|L_{s.i.}|$', fontsize = 14)

    plt.legend(['Unstable', 'Stable'], fontsize = 14)
    plt.tight_layout()

    #plt.savefig(r'C:\Users\Korisnik\OneDrive\Desktop\3-body\M1 M2 SLICES, PCT\PLOT OF ABS_L VS T FOR M1 BIGGER THAN M2, PCT_{0}.pdf'.format(pct))
    plt.close()
    
    
#%% PLOTTING SLICES IN M1 M2 SPACE OF T_S.I. VS |L_S.I.| FOR M1 LESS THAN M2


PCT = [1.00 + i*0.05 for i in range(5)]

for pct in PCT:
    
    pdf_stable = df_stable_2[ df_stable_2['m_2'].between(pct*df_stable_2['m_1']-0.001, pct*df_stable_2['m_1']+0.001) ]
    pdf_unstable = df_unstable_2[ df_unstable_2['m_2'].between(pct*df_unstable_2['m_1']-0.001, pct*df_unstable_2['m_1']+0.001) ]
    
    
    plt.scatter( pdf_unstable['T_E']/pdf_unstable['M_tot']**(5/2), abs(pdf_unstable['L_E'])/pdf_unstable['M_tot']**(13/6), label = 'Unstable')
    plt.scatter( pdf_stable['T_E']/pdf_stable['M_tot']**(5/2), abs(pdf_stable['L_E'])/pdf_stable['M_tot']**(13/6), label = 'Stable')


    plt.xlabel('$T_{s.i.}$', fontsize = 14)
    plt.ylabel('$|L_{s.i.}|$', fontsize = 14)

    plt.legend(['Unstable', 'Stable'], fontsize = 14)
    plt.tight_layout()

    #plt.savefig(r'C:\Users\Korisnik\OneDrive\Desktop\3-body\M1 M2 SLICES, PCT\PLOT OF ABS_L VS T FOR M1 LESS THAN M2, PCT_{0}.pdf'.format(pct))
    plt.close()








#%% IMPORTING LIBRARY FOR ANIMATION

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
    
#%% ANIMATION OF |L*| VS M_TOT FOR M1 = pct*M2 \pm 0.001 pct >= 1

PCTs = np.array( [1+i*0.001 for i in range(300)] )

fig, ax = plt.subplots()



def animate(i):
    ax.clear()
   
    pct = PCTs[i]
   
    ax.set_xlim( [7, 12.5] )
    ax.set_ylim( [-0.005, 0.065] )
    ax.set_xlabel('$M_{tot}^{13/6}$', fontsize = 14, labelpad = 1)
    ax.set_ylabel('$|L^*|$', fontsize = 14)
    
    pdf_stable = df_stable_1[ df_stable_1['m_1'].between(pct*df_stable_1['m_2']-0.001, pct*df_stable_1['m_2']+0.001) ]
    pdf_unstable = df_unstable_1[ df_unstable_1['m_1'].between(pct*df_unstable_1['m_2']-0.001, pct*df_unstable_1['m_2']+0.001) ]

    ax.scatter(pdf_unstable['M_tot']**(13/6), abs(pdf_unstable['L_E']), label = 'Unstable')
    ax.scatter(pdf_stable['M_tot']**(13/6), abs(pdf_stable['L_E']), label = 'Stable')
    
    ax.legend(fontsize = 14)
    

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(PCTs) - 1, interval=1)


#ani.save(r"C:\Users\Korisnik\OneDrive\Desktop\3-body\ANIMATION OF ABS_L VS M_TOT FOR M1 BIGGER THAN M2.gif", writer=PillowWriter(fps=15))


#%% ANIMATION OF T* VS M_TOT FOR M1 = pct*M2 \pm 0.001 pct >= 1

PCTs = np.array( [1+i*0.001 for i in range(300)] )

fig, ax = plt.subplots()



def animate(i):
    ax.clear()
   
    pct = PCTs[i]
   
    ax.set_xlim( [10, 19] )
    ax.set_ylim( [14, 30] )
    ax.set_xlabel('$M_{tot}^{5/2}$', fontsize = 14, labelpad = 1)
    ax.set_ylabel('$T^*$', fontsize = 14)

    
    pdf_stable = df_stable_1[ df_stable_1['m_1'].between(pct*df_stable_1['m_2']-0.001, pct*df_stable_1['m_2']+0.001) ]
    pdf_unstable = df_unstable_1[ df_unstable_1['m_1'].between(pct*df_unstable_1['m_2']-0.001, pct*df_unstable_1['m_2']+0.001) ]

    ax.scatter(pdf_unstable['M_tot']**(5/2), pdf_unstable['T_E'], label = 'Unstable')
    ax.scatter(pdf_stable['M_tot']**(5/2), pdf_stable['T_E'], label = 'Stable')
    
    ax.legend(loc = 'upper left', fontsize = 14)
    

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(PCTs) - 1, interval=1)


#ani.save(r"C:\Users\Korisnik\OneDrive\Desktop\3-body\ANIMATION OF T VS M_TOT FOR M1 BIGGER THAN M2.gif", writer=PillowWriter(fps=15))




#%% ANIMATION OF T* VS |L*| FOR M1 = pct*M2 \pm 0.001 pct >= 1

PCTs = np.array( [1+i*0.001 for i in range(300)] )

fig, ax = plt.subplots()



def animate(i):
    ax.clear()
   
    pct = PCTs[i]
    
    ax.set_xlim( [1.5, 1.6] )
    ax.set_ylim( [-0.001, 0.008] )
    ax.set_xlabel('$T^* / M_{tot}^{5/2}$', fontsize = 14, labelpad = 1)
    ax.set_ylabel('$|L^*| / M_{tot}^{13/6}$', fontsize = 14, labelpad = -8)

    
    pdf_stable = df_stable_1[ df_stable_1['m_1'].between(pct*df_stable_1['m_2']-0.001, pct*df_stable_1['m_2']+0.001) ]
    pdf_unstable = df_unstable_1[ df_unstable_1['m_1'].between(pct*df_unstable_1['m_2']-0.001, pct*df_unstable_1['m_2']+0.001) ]

    ax.scatter( pdf_unstable['T_E']/pdf_unstable['M_tot']**(5/2), abs(pdf_unstable['L_E'])/pdf_unstable['M_tot']**(13/6), label = 'Unstable')
    ax.scatter( pdf_stable['T_E']/pdf_stable['M_tot']**(5/2), abs(pdf_stable['L_E'])/pdf_stable['M_tot']**(13/6), label = 'Stable')
    
    ax.legend(loc = 'upper right', fontsize = 14)
    

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(PCTs) - 1, interval=1)


#ani.save(r"C:\Users\Korisnik\OneDrive\Desktop\3-body\ANIMATION OF T VS abs_L FOR M1 BIGGER THAN M2.gif", writer=PillowWriter(fps=15))



#%% ANIMATION OF L* VS M_TOT FOR M2 = pct*M1 \pm 0.001 pct >= 1

PCTs = np.array( [1+i*0.001 for i in range(300)] )

fig, ax = plt.subplots()



def animate(i):
    ax.clear()
   
    pct = PCTs[i]
   
    ax.set_xlim( [7.5, 13.5] )
    ax.set_ylim( [-0.005, 0.055] )
    ax.set_xlabel('$M_{tot}^{13/6}$', fontsize = 14, labelpad = 1)
    ax.set_ylabel('$|L^*|$', fontsize = 14)
    
    pdf_stable = df_stable_2[ df_stable_2['m_2'].between(pct*df_stable_2['m_1']-0.001, pct*df_stable_2['m_1']+0.001) ]
    pdf_unstable = df_unstable_2[ df_unstable_2['m_2'].between(pct*df_unstable_2['m_1']-0.001, pct*df_unstable_2['m_1']+0.001) ]

    ax.scatter(pdf_unstable['M_tot']**(13/6), abs(pdf_unstable['L_E']), label = 'Unstable')
    ax.scatter(pdf_stable['M_tot']**(13/6), abs(pdf_stable['L_E']), label = 'Stable')
    
    ax.legend(fontsize = 14)

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(PCTs) - 1, interval=1)

#ani.save(r"C:\Users\Korisnik\OneDrive\Desktop\3-body\ANIMATION OF ABS_L VS M_TOT FOR M1 LESS THAN M2.gif", writer=PillowWriter(fps=15))


#%% ANIMATION OF T* VS M_TOT FOR M2 = pct*M1 \pm 0.001 pct >= 1

PCTs = np.array( [1+i*0.001 for i in range(300)] )

fig, ax = plt.subplots()



def animate(i):
    ax.clear()
   
    pct = PCTs[i]
   
    ax.set_xlim( [10, 21] )
    ax.set_ylim( [14, 32] )
    ax.set_xlabel('$M_{tot}^{5/2}$', fontsize = 14, labelpad = 1)
    ax.set_ylabel('$T^*$', fontsize = 14)
    
    pdf_stable = df_stable_2[ df_stable_2['m_2'].between(pct*df_stable_2['m_1']-0.001, pct*df_stable_2['m_1']+0.001) ]
    pdf_unstable = df_unstable_2[ df_unstable_2['m_2'].between(pct*df_unstable_2['m_1']-0.001, pct*df_unstable_2['m_1']+0.001) ]

    ax.scatter(pdf_unstable['M_tot']**(5/2), pdf_unstable['T_E'], label = 'Unstable')
    ax.scatter(pdf_stable['M_tot']**(5/2), pdf_stable['T_E'], label = 'Stable')
    
    ax.legend(loc = 'upper left', fontsize = 14)
    

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(PCTs) - 1, interval=1)


#ani.save(r"C:\Users\Korisnik\OneDrive\Desktop\3-body\ANIMATION OF T VS M_TOT FOR M1 LESS THAN M2.gif", writer=PillowWriter(fps=15))



#%% ANIMATION OF T* VS |L*| FOR M2 = pct*M1 \pm 0.001 pct >= 1

PCTs = np.array( [1+i*0.001 for i in range(300)] )

fig, ax = plt.subplots()



def animate(i):
    ax.clear()
   
    pct = PCTs[i]
    
    ax.set_xlim( [1.43, 1.58] )
    ax.set_ylim( [-0.001, 0.004] )
    ax.set_xlabel('$T^* / M_{tot}^{5/2}$', fontsize = 14, labelpad = 1)
    ax.set_ylabel('$|L^*| / M_{tot}^{13/6}$', fontsize = 14, labelpad = -8)

    
    pdf_stable = df_stable_2[ df_stable_2['m_2'].between(pct*df_stable_2['m_1']-0.001, pct*df_stable_2['m_1']+0.001) ]
    pdf_unstable = df_unstable_2[ df_unstable_2['m_2'].between(pct*df_unstable_2['m_1']-0.001, pct*df_unstable_2['m_1']+0.001) ]

    ax.scatter( pdf_unstable['T_E']/pdf_unstable['M_tot']**(5/2), abs(pdf_unstable['L_E'])/pdf_unstable['M_tot']**(13/6), label = 'Unstable')
    ax.scatter( pdf_stable['T_E']/pdf_stable['M_tot']**(5/2), abs(pdf_stable['L_E'])/pdf_stable['M_tot']**(13/6), label = 'Stable')
    
    ax.legend(loc = 'upper right', fontsize = 14)
    

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(PCTs) - 1, interval=1)


#ani.save(r"C:\Users\Korisnik\OneDrive\Desktop\3-body\ANIMATION OF T VS abs_L FOR M1 LESS THAN M2.gif", writer=PillowWriter(fps=15))














