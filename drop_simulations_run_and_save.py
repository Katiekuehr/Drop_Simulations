"""
Importing necessary packages
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scipy.special import eval_legendre
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import os
import csv


def Legendre(x, n, n_thetas): #Define Legendre Polynomial of cosine
    """
    Function that evaluates nth Legendre polynomial at x with a scipy function eval_legendre.

    Inputs:
    x (float): X-value at which Legendre polynomial is evaluated
    n (int): Number of Legendre polynomial to be evaluated
    n_thetas (int): Total number of Legendre Polynomials used.

    Output:
    value (float): value of nth Legendre polynomial at x

    """
    if n == n_thetas and x != 1:
        #To avoid numerical errors very close to zero
        value = 0
    else:
      value = eval_legendre(n, x)
    return value

def matrix(delta_t,q, n_thetas, cos_theta_vec, Oh): #Define function for matrix

    """
    Function that creates the matrix with a given time step and a given number of conact points.
    
    Inputs:
    delta_t (float): Chosen time-step-size.
    q (int): Number of guessed contact points.
    n_thetas (int): Total number of sampled angles.
    cos_theta_vec (vector): Vector of cosins of all sampled angles theta.

    Output:
    M_mat (matrix): Filled matrix with q guessed contact points.

    """
    #First Block Row
    #Identity Matrix I(n_thetas -1)
    Identity_n_thetas_min_1_mat = np.eye((n_thetas-1)) #Condition Number = 1

    M_mat = np.concatenate((Identity_n_thetas_min_1_mat,
                            -delta_t*Identity_n_thetas_min_1_mat,
                            np.zeros((n_thetas-1,n_thetas+1)),
                            np.zeros((n_thetas-1,1)),
                            np.zeros((n_thetas-1,1))
                           ),
                           1
                          )
    #Second Block Row
    C_list = [((l - 1) * (l) * (l + 2)) for l in range(2, n_thetas+1)]
    C_mat = np.diag(C_list)
    D_list = [1 + delta_t*2*Oh*(l-1)*(2*l+1) for l in range(2, n_thetas+1)]
    D_mat = np.diag(D_list)
    E_list = [l for l in range(2, n_thetas+1)]
    E_mat = np.concatenate((np.zeros((n_thetas-1, 2)),
                            np.diag(E_list)),1)
    second_block_row = np.concatenate((delta_t*C_mat,
                                            D_mat,
                                            delta_t*E_mat,
                                            np.zeros((n_thetas-1,1)),
                                            np.zeros((n_thetas-1,1))
                                           ),
                                           1
                                          )
    M_mat = np.concatenate((M_mat,
                            second_block_row),
                           0
                          )
    #Third and fourth rows of blocks
    FH_mat = np.zeros((n_thetas+1,n_thetas+1))
    for ind_angle in range(n_thetas+1):
      for ind_poly in range(n_thetas+1):
        FH_mat[ind_angle,ind_poly] = Legendre(cos_theta_vec[ind_angle],ind_poly, n_thetas)
    F_mat = FH_mat[0:q,2:]
    H_mat = FH_mat[q:,:]
    G_vec = np.transpose([-1/cos_theta_vec[0:q]])
    third_block_row = np.concatenate((F_mat,
                                      np.zeros((q,n_thetas-1)),
                                      np.zeros((q,n_thetas+1)),
                                      G_vec,
                                      np.zeros((q,1))
                                     ),
                                     1
                                    )
    fourth_block_row = np.concatenate((np.zeros((n_thetas+1-q,n_thetas-1)),
                                       np.zeros((n_thetas+1-q,n_thetas-1)),
                                       H_mat,
                                       np.zeros((n_thetas+1-q,1)),
                                       np.zeros((n_thetas+1-q,1))
                                      ),
                                      1
                                     )
    M_mat = np.concatenate((M_mat,
                            third_block_row,
                            fourth_block_row
                           ),
                           0
                          )
    #fifth block row
    fifth_block_row = np.concatenate((np.zeros((1,n_thetas-1)),
                                      np.zeros((1,n_thetas-1)),
                                      np.zeros((1,n_thetas+1)),
                                      [[1]],
                                      [[-delta_t]]
                                     ),
                                     1
                                    )
    M_mat = np.concatenate((M_mat,
                            fifth_block_row
                           ),
                           0
                          )
    #Sixth row of blocks
    H = np.zeros((1,n_thetas+1))
    H[0,1] = -1
    sixth_block_row = np.concatenate((np.zeros((1,n_thetas-1)),
                                      np.zeros((1,n_thetas-1)),
                                      delta_t*H,
                                      [[0]],
                                      [[1]]
                                     ),
                                     1
                                    )
    M_mat = np.concatenate((M_mat,
                            sixth_block_row
                           ),
                           0
                          )
    return M_mat

def error(amplitudes_vec, h, q, n_thetas, cos_theta_vec): #Defining the Error function

    """
    Function that calculates the error of q contact points.

    Inputs:
    amplitudes_vec (vector): Vector of amplitudes for q contact points at k+1.
    h (float): Height of the center of mass at k+1.
    q (int): Number of guessed contact points.
    n_thetas (int): Total number of sampled angles.
    cos_theta_vec (vector): Vector of cosins of all sampled angles theta.

    Outputs:
    err (float): Error at k+1 with q contact points.
    """

    if q < 0: #Handling impossible situation
        return np.inf

    if q == 0:
      all_verticals_list = []
      for ind_theta in range(n_thetas+1):
          cos_theta = cos_theta_vec[ind_theta]
          height = ((1
                     + np.sum([amplitudes_vec[l]*Legendre(cos_theta, l+2, n_thetas) for l in range(n_thetas-1)])
                    )
                    *cos_theta
                   )
          all_verticals_list.append(height)

      for vertical in all_verticals_list: #Checking whether droplet crosses surface (physically impossible)
        if vertical > h:
          return np.inf
      return 0 #Else, not in contact

    #Variables to store sums
    sum_q = 1
    sum_qp1 = 1

    #Extracting the cos(theta) values at q (= q-1 in list index) and q+1 (= q in list index)
    cos_theta_q = cos_theta_vec[q-1]
    cos_theta_q_plus_1 = cos_theta_vec[q]

    for ind_poly in range(n_thetas-1):
      sum_q += (amplitudes_vec[ind_poly]
                *Legendre(cos_theta_q,ind_poly+2, n_thetas)
               )
      sum_qp1 += (amplitudes_vec[ind_poly]
                  *Legendre(cos_theta_q_plus_1,ind_poly+2, n_thetas)
                 )
    #Calculating the vertical components of each point
    vertical_q = sum_q * cos_theta_q
    vertical_qp1 = sum_qp1 * cos_theta_q_plus_1

    #Regular error calculation
    err1 = np.abs(vertical_q - vertical_qp1)
    err2 = 0 #Err2 for checking whether droplet crosses surface (physically impossible)
    for ind_theta in range(q,n_thetas+1):
      rad = 1
      for ind_poly in range(2,n_thetas+1):
        rad += (amplitudes_vec[ind_poly-2]
                *Legendre(cos_theta_vec[ind_theta],ind_poly, n_thetas)
               )
      if rad*cos_theta_vec[ind_theta]>(h):
        err2 = np.inf
        break
    err = np.max([err1,[err2]]) #Return highest error

    return err

def try_q(amps_prev_vec, amps_vel_prev_vec, h_prev, v_prev, q, delta_t, n_thetas, Bo, Oh):

  """
  Function that "tries" q contact points by solving the system of equations.

  Inputs:
  amps_prev_vec (vector): Vector of amplitudes at the previous time (k).
  amps_vel_prev_vec (vector): Vector of velocities of amplitudes at the previous time (k).
  h_prev (float): Height of the center of mass at the previous time (k).
  v_prev (float): Velocity of the center of mass at the previous time (k).
  delta_t (float): Time step (in non-dimensional time).
  q (int): Number of guessed contact points.
  n_thetas (int): Total number of sampled angles.
  Bo (float): Bond (non-dimensional gravity).

  Outputs:
  sol_vec (vector): Vector of the solution of matrix at q contact points at time k+1
  err (float): Error at k+1 with q contact points.
  
  """
  #Extracting samling angles
  roots, weights = roots_legendre(n_thetas)
  cos_theta_vec = np.concatenate(([1], np.flip(roots)))
  q_last = np.argmax(cos_theta_vec <= 0)

  #Running through different guesses of q
  if q < 0: #Return previous matrix and infinite error for physically impossible case
    sol_vec = np.concatenate((amps_prev_vec,
                              amps_vel_prev_vec,
                              np.zeros((n_thetas+1,1)),
                              h_prev,
                              v_prev
                             ),
                             0
                            )
    err = np.inf

  elif q == 0: #Only solve necessary parts of the matrix (no contact and pressure block-rows)
    #Right-hand side for mode equations
    RHS_modes_vec = np.concatenate((amps_prev_vec,
                                    amps_vel_prev_vec
                                   ),
                                   0
                                  )
    #right-hand side for cm equations
    RHS_h_vec = np.concatenate((h_prev,
                                v_prev-(delta_t*Bo)
                               ),
                               0
                              )

    #Extracting parts of the matrices for modes and cm
    M_modes_mat = matrix(delta_t, q, n_thetas, cos_theta_vec, Oh)[: int(n_thetas*2-2), : int(n_thetas*2-2)]
    M_h_mat = matrix(delta_t, q, n_thetas, cos_theta_vec, Oh)[-2:,-2:]

    #Solving systems of equations
    sol_modes_vec = solve(M_modes_mat, RHS_modes_vec)
    sol_h_vec = solve(M_h_mat, RHS_h_vec)
    sol_vec = np.concatenate((sol_modes_vec,
                              np.zeros((n_thetas+1,1)),
                              sol_h_vec
                             ),
                            0
                           )

    #Calculating the error at k+1, q=0
    err = error(sol_modes_vec[: n_thetas-1],
                sol_h_vec[0],
                q,
                n_thetas,
                cos_theta_vec)

  elif q <= q_last: #Solving the full matrix with guessed contact points
    RHS_vec = np.concatenate((amps_prev_vec,
                              amps_vel_prev_vec,
                              -np.ones((q,1)),
                              np.zeros((n_thetas+1-q,1)),
                              h_prev,
                              v_prev-(delta_t*Bo)
                             ),
                             0
                            )
    M_mat = (matrix(delta_t, q, n_thetas, cos_theta_vec, Oh))
    sol_vec = (solve(M_mat, RHS_vec))

    #Calculating the error at k+1, q
    err = error(sol_vec[: n_thetas-1], sol_vec[-2], q, n_thetas, cos_theta_vec)

  else: #Assumptions of little deformation in the model do not allow for this case; returns sample matrix and infinite error
    sol_vec = np.concatenate((amps_prev_vec,
                              amps_vel_prev_vec,
                              np.zeros((n_thetas+1,1)),
                              h_prev,
                              v_prev
                             ),
                             0
                            )
    err = np.inf

  return (sol_vec, err)

def north_and_south_pole(all_amps_vec, h, n_thetas):

    """
    Function that evaluates the deformed droplet's north and south pole radii.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at a single point in time.
    h (float): Height of the center of mass.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    deformed_north_pole_radius (vector): Radius at the north pole.
    deformed_south_pole_radius (vector): Radius at the south pole.

    """
    #Setting undeformed radii
    undeformed_north_pole_radius = 1
    undeformed_south_pole_radius = 1

    #Adding deformations from the Legendre modes
    for i in range(2, n_thetas):
        undeformed_north_pole_radius += all_amps_vec[i-2] * Legendre(-1, i, n_thetas)
        undeformed_south_pole_radius += all_amps_vec[i-2] * Legendre(1, i, n_thetas) 

    #Converting to radians to convert to xy-coordinates/ height
    radian_north_pole = (np.pi)/2
    radian_south_pole = (np.pi*3)/2 

    #Obtaining xy coordinates
    deformed_north_pole_radius = undeformed_north_pole_radius * np.sin(radian_north_pole) + h
    deformed_south_pole_radius = undeformed_south_pole_radius * np.sin(radian_south_pole) + h

    return[deformed_north_pole_radius, deformed_south_pole_radius]

def maximum_radius(all_amps_vec, n_thetas):

    """
    Function that calculates the maximum radius of the droplet at a single point in time.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at a single point in time.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    radius_of_largest_deformation (vector): Radius of the angle, where largest radius is achieved.

    """

    #Extracting samling angles
    roots, weights = roots_legendre(n_thetas)
    cos_theta_vec = np.concatenate(([1], np.flip(roots)))
    theta_vec = np.arccos(cos_theta_vec)

    radii = np.ones(n_thetas) #Setting radi at all sampled points to an undeformed radius
    for i in range(2, n_thetas):
        for ind_angles in range(n_thetas): #Adding deformation
            radii[ind_angles] += all_amps_vec[i - 2] * Legendre(np.cos(theta_vec[ind_angles]), i, n_thetas)

    radius_of_largest_deformation = np.max(radii)
    return radius_of_largest_deformation #Returning largest radius

def coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_times_list, Bo, We):

    """
    Function that calculates the coefficient of restitution and contact time for a droplet impact.

    Inputs:
    all_h_list (list): List of heights of the center of mass of the droplet at every time step.
    all_v_list (list): List of velocities of the center of mass of the droplet at every time step.
    all_m_list (list): List of contact points at every time step.
    all_time_list (list): List of all time steps.
    Bo (float): Bond Number, non-dimensional gravity.
    We (float): Weber Number, non-dimensional velocity.

    Outputs:
    coeff_rest_squared_float (float): Coefficient of Restitution.
    contact_time_float (float): Time of contact (non-dimensional).

    """

    #Setting all values to be obtained to None to avoid errors
    final_velocity = None
    final_height = None
    contact_time_float = False
    index = None
    
    for i in range(1, len(all_m_list)-1): 
        if all_m_list[i] == 0 and all_m_list[i-1] > 0: #Marks end of contact
            index = i
            break

    #Averages of last contact and first non-comtact times
    final_velocity = (all_v_list[index] + all_v_list[index-1])/2
    final_height = (all_h_list[index] + all_h_list[index-1])/2
    contact_time_float = (all_times_list[index] + all_times_list[index-1])/2

    #Handling if no take-off occurs
    if final_velocity == None:
        return None

    #Calculating of coefficient of restitution squared
    coeff_rest_squared = (((Bo * (final_height - 1)) 
                          + 0.5 * (final_velocity**2))/ 
                          (0.5 * (We)))
    
    coeff_rest_squared_float = np.sqrt(np.abs(float(coeff_rest_squared)))


    return [coeff_rest_squared_float, contact_time_float]

def maximum_contact_radius(all_amps_list, all_m_list, n_thetas):

    """
    Function that calculates the maximum contact radius of a droplet impact.

    Inputs:
    all_amps_list (list): List of all amplitudes at every time step.
    all_m_list (list): List of contact points at every time step.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    np.max(contact_radii) (float): Maximum contact radius achieved during impact.

    """
    #Extracting samling angles
    roots, weights = roots_legendre(n_thetas)
    cos_theta_vec = np.concatenate(([1], np.flip(roots)))
    theta_vec = np.arccos(cos_theta_vec)

    contact_radii = []
    new_aplitudes = np.asarray(all_amps_list) #Converting to array for easier handling
    for time in range(len(all_amps_list)):
        m = all_m_list[time]
        radius = np.ones(2)
        if m == 0: #If not in contact, contact radius = 0
            contact_radii.append(0)
        else:
            for i in range(2, n_thetas):
                radius[0] += new_aplitudes[time, i - 2] * Legendre(np.cos(theta_vec[m-1]), i, n_thetas)
                radius[1] += new_aplitudes[time, i - 2] * Legendre(np.cos(theta_vec[m]), i, n_thetas)

            new_thetas = [theta + 3 * np.pi / 2 for theta in theta_vec]
            x_1 = radius[0] * np.cos(new_thetas[m-1])
            x_2 = radius[1] * np.cos(new_thetas[m])

            #Calculating average of last contact-point-radius and first non-contact-point radius
            contact_radii.append((x_1+x_2)/2)

    return np.max(contact_radii) #Returning largest contact area

def running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo, Oh):

    """
    Function that runs the simulation of an impacting droplet.

    Inputs:
    n_thetas (int): Total number of sampled angles.
    n_sampling_time_L_mode
    T_end (int): Duration of simulation.
    H (float): Initial height.
    V (float): Initial velocity.
    Bo (float): Bond Number, non-dimensional gravity.

    Outputs:
    all_amps_list (list): List of all amplitudes at all time steps.
    all_amps_vel_list (list): List of all amplitude velocities at all time steps.
    all_press_list (list):  List of all pressure amplitudes at all time steps.
    all_h_list (list): List of heights of the center of mass of the droplet at every time step.
    all_v_list (list): List of velocities of the center of mass of the droplet at every time step.
    all_m_list (list): List of contact points at every time step.
    all_north_poles_list (list): List of north poles at every time step.
    all_south_poles_list (list): List of south poles at every time step.
    all_maximum_radii_list (list): List of maximum radii at every time step.
    all_times_list (list): List of all time steps.

    """

    ind_time = 0

    #Numerical constants
    delta_time_max = (2*np.pi
                    /(np.sqrt(n_thetas*(n_thetas+2)*(n_thetas-1))
                        *n_sampling_time_L_mode))
    n_times_base = int(np.ceil(T_end/delta_time_max))

    #Initialize list of times
    all_times_list = (np.linspace(0,n_times_base,n_times_base+1)
                * delta_time_max).tolist()
    
    #main time loop
    amps_prev_vec = np.zeros((n_thetas-1,1))
    amps_vel_prev_vec = np.zeros((n_thetas-1,1))
    h_prev = H*np.ones((1,1))
    v_prev = V*np.ones((1,1))
    m_prev = 0

    #Variables to store values of simulation
    all_amps_list = [amps_prev_vec]
    all_amps_vel_list = [amps_vel_prev_vec]
    all_press_list = [np.zeros((n_thetas+1,1))]
    all_h_list = [h_prev]
    all_v_list = [v_prev]
    all_m_list = [0]
    all_north_poles_list = [north_and_south_pole(amps_prev_vec, H, n_thetas)[0]]
    all_south_poles_list = [north_and_south_pole(amps_prev_vec, H, n_thetas)[1]]
    all_maximum_radii_list = [maximum_radius(amps_prev_vec, n_thetas)]
    
    
    while all_times_list[ind_time] < all_times_list[-1]:

        #Time step is the difference in the times_list
        delta_t = all_times_list[ind_time+1] - all_times_list[ind_time]

        #Set all errors to infinity
        err_m_prev = np.inf
        err_m_prev_m1 = np.inf
        err_m_prev_m2 = np.inf
        err_m_prev_p1 = np.inf
        err_m_prev_p2 = np.inf

        #Calculate solution vectors and errors for initial comparison (q=q_{k-1}; q=q_{k-1}-1)
        sol_vec_m_prev, err_m_prev = try_q(amps_prev_vec,
                                        amps_vel_prev_vec,
                                        h_prev,
                                        v_prev,
                                        m_prev,
                                        delta_t,
                                        n_thetas,
                                        Bo,
                                        Oh)
        sol_vec_m_prev_m1, err_m_prev_m1 = try_q(amps_prev_vec,
                                                amps_vel_prev_vec,
                                                h_prev,
                                                v_prev,
                                                m_prev-1,
                                                delta_t,
                                                n_thetas,
                                                Bo,
                                                Oh)                                       

        #If one contact point less is better than the previous contact points, test two contact points less (q=q_{k-1}-2)
        if err_m_prev_m1 < err_m_prev:
            sol_vec_m_prev_m2, err_m_prev_m2 = try_q(amps_prev_vec,
                                                    amps_vel_prev_vec,
                                                    h_prev,
                                                    v_prev,
                                                    m_prev-2,
                                                    delta_t,
                                                    n_thetas, 
                                                    Bo,
                                                    Oh)
            if err_m_prev_m2 < err_m_prev_m1:

                #If two contact points less were better, the time step was too large: Insert new time step
                time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1])/2
                all_times_list.insert(ind_time+1, time_insert)

            else:
                #Accept solution, update variables, append to lists, and move forward in time
                ind_time += 1
                m_prev -= 1
                amps_prev_vec = sol_vec_m_prev_m1[:(n_thetas-1)]
                amps_vel_prev_vec = sol_vec_m_prev_m1[(n_thetas-1):(2*n_thetas-2)]
                press_prev_vec = sol_vec_m_prev_m1[(2*n_thetas-2):-2]
                h_prev = np.reshape(sol_vec_m_prev_m1[-2],(1,1))
                v_prev = np.reshape(sol_vec_m_prev_m1[-1],(1,1))
                all_amps_list.append(amps_prev_vec)
                all_amps_vel_list.append(amps_vel_prev_vec)
                all_press_list.append(press_prev_vec)
                all_h_list.append(h_prev)
                all_v_list.append(v_prev)
                all_m_list.append(m_prev)
                all_north_poles_list.append(north_and_south_pole(amps_prev_vec, h_prev, n_thetas)[0])
                all_south_poles_list.append(north_and_south_pole(amps_prev_vec, h_prev, n_thetas)[1])
                all_maximum_radii_list.append(maximum_radius(amps_prev_vec, n_thetas))

        else: #If one contact point less is not better, try one contact point (more q=q_{k-1}+1)
            sol_vec_m_prev_p1, err_m_prev_p1 = try_q(amps_prev_vec,
                                                    amps_vel_prev_vec,
                                                    h_prev,
                                                    v_prev,
                                                    m_prev+1,
                                                    delta_t,
                                                    n_thetas,
                                                    Bo, 
                                                    Oh)
        if err_m_prev_p1 < err_m_prev: 
            sol_vec_m_prev_p2, err_m_prev_p2 = try_q(amps_prev_vec,
                                                    amps_vel_prev_vec,
                                                    h_prev,
                                                    v_prev,
                                                    m_prev+2,
                                                    delta_t,
                                                    n_thetas,
                                                    Bo,
                                                    Oh)
            if err_m_prev_p2 < err_m_prev_p1: #If one contact point more is better than the previous contact points, test two contact points more (q=q_{k-1}+2)
            #If two contact points less were better, the time step was too large: Insert new time step
                time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1])/2
                all_times_list.insert(ind_time+1, time_insert)
            else:
                #Accept solution, update variables, append to lists, and move forward in time
                ind_time += 1
                m_prev += 1
                amps_prev_vec = sol_vec_m_prev_p1[:(n_thetas-1)]
                amps_vel_prev_vec = sol_vec_m_prev_p1[(n_thetas-1):(2*n_thetas-2)]
                press_prev_vec = sol_vec_m_prev_p1[(2*n_thetas-2):-2]
                h_prev = np.reshape(sol_vec_m_prev_p1[-2],(1,1))
                v_prev = np.reshape(sol_vec_m_prev_p1[-1],(1,1))
                all_amps_list.append(amps_prev_vec)
                all_amps_vel_list.append(amps_vel_prev_vec)
                all_press_list.append(press_prev_vec)
                all_h_list.append(h_prev)
                all_v_list.append(v_prev)
                all_m_list.append(m_prev)
                all_north_poles_list.append(north_and_south_pole(amps_prev_vec, h_prev, n_thetas)[0])
                all_south_poles_list.append(north_and_south_pole(amps_prev_vec, h_prev, n_thetas)[1])
                all_maximum_radii_list.append(maximum_radius(amps_prev_vec, n_thetas))
            
        elif err_m_prev_p1 == np.inf and err_m_prev == np.inf and err_m_prev_m1 == np.inf:
            #If all errors are infinite, reduce the time step
            time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1])/2
            all_times_list.insert(ind_time+1, time_insert)

        else: 
            #Accept solution (unchanged q), update variables, append to lists, and move forward in time
            ind_time += 1
            amps_prev_vec = sol_vec_m_prev[:(n_thetas-1)]
            amps_vel_prev_vec = sol_vec_m_prev[(n_thetas-1):(2*n_thetas-2)]
            press_prev_vec = sol_vec_m_prev[(2*n_thetas-2):-2]
            h_prev = np.reshape(sol_vec_m_prev[-2],(1,1))
            v_prev = np.reshape(sol_vec_m_prev[-1],(1,1))
            all_amps_list.append(amps_prev_vec)
            all_amps_vel_list.append(amps_vel_prev_vec)
            all_press_list.append(press_prev_vec)
            all_h_list.append(h_prev)
            all_v_list.append(v_prev)
            all_m_list.append(m_prev)
            all_north_poles_list.append(north_and_south_pole(amps_prev_vec, h_prev, n_thetas)[0])
            all_south_poles_list.append(north_and_south_pole(amps_prev_vec, h_prev, n_thetas)[1])
            all_maximum_radii_list.append(maximum_radius(amps_prev_vec, n_thetas))

    return [all_amps_list, all_amps_vel_list, all_press_list, all_h_list, all_v_list, all_m_list, all_north_poles_list, all_south_poles_list, all_maximum_radii_list, all_times_list]

def height_plots(all_north_poles_list, all_south_poles_list, all_h_list, all_times_list, folder_name):

    """
    Function that creates a plot of all north pole, south pole and center of mass heights and saves it to a designated folder.

    Inputs:
    all_north_poles_list (list): List of north poles at every time step.
    all_south_poles_list (list): List of south poles at every time step.
    all_h_list (list): List of heights of the center of mass of the droplet at every time step.
    all_times_list (list): List of all time steps.
    folder_name (string): Name of folder on the computer where the simulation should be saved to.

    Outputs:
    plot_path (saved plot): Saves plot on computer.
    
    """

    #Convert arrays within storage lists to simple floats for plotting
    all_north_poles_float_list = [arr.item() for arr in all_north_poles_list]
    all_south_poles_float_list = [arr.item() for arr in all_south_poles_list]
    all_h_float_list = [arr.item() for arr in all_h_list]

    #Plot
    plt.figure()
    plt.plot(all_times_list, all_north_poles_float_list, label="North Pole", color="blue")
    plt.plot(all_times_list, all_south_poles_float_list, label="South Pole", color="red")
    plt.plot(all_times_list, all_h_float_list, label="Center of Mass", color="green")

    #Labels
    plt.legend()
    plt.xlabel("Non-Dimensional Time")
    plt.ylabel("North/ South Pole/ Center of Mass in Radii")

    #Saving to designated path on computer
    plot_path = os.path.join(folder_name, 'height_plots.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def maximum_radius_plot(all_maximum_radii_list, all_times_list, folder_name):

    """
    Function that creates a plot of the maximum droplet radius over time and saves it to a designated path on your computer. 

    Inputs:
    all_maximum_radii_list (list): List of maximum radii at every time step.
    all_times_list (list): List of all time steps.
    folder_name (string): Name of folder on the computer where the simulation should be saved to.

    Outputs:
    plot_path (saved plot): Saves plot on computer.

    """

    #Plot
    plt.figure()
    plt.plot(all_times_list, all_maximum_radii_list)

    #Labels
    plt.ylabel("Maxmum Radius in Non-Dimensional Radii")
    plt.xlabel("Non-Dimensional Time")
    plt.legend()

    #Saving to designated path on computer
    plot_path = os.path.join(folder_name, 'max_radius_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def amplitudes_over_time_plot(all_amps_list, all_times_list, folder_name, n_thetas):

    """
    Function that creates a plot of all amplitudes of all Legendre modes modelling the deformation of the droplet over time 
        and saves it to a designated path on your computer. 

    Inputs:
    all_amps_list (list): List of all amplitudes at all time steps.
    all_times_list (list): List of all time steps.
    folder_name (string): Name of folder on the computer where the simulation should be saved to.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    plot_path (saved plot): Saves plot on computer.

    """

    #Convert to array for easier handling
    amplitudes_array = np.asarray(all_amps_list)

    #Initialize plot and colormap
    fig, ax = plt.subplots()
    colors = cm.viridis(np.linspace(0, 1, n_thetas))

    #Plot individual amplitudes with designated colors
    for amp_ind in range(n_thetas-1):
        coefficients_list = []
        for time_ind in range(len(amplitudes_array)):
            coefficients_list.append(amplitudes_array[time_ind, amp_ind])
        ax.plot(all_times_list, coefficients_list, color=colors[amp_ind])
    
    #Labels
    plt.ylabel("Amplitude")
    plt.xlabel("Non-Dimensional Time")

    #Create a ScalarMappable and add a colorbar
    norm = mcolors.Normalize(vmin=2, vmax=n_thetas)
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    
    #Creating Colorbar 
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([2, n_thetas])
    cbar.set_ticklabels([2, n_thetas])
    cbar.set_label('Amplitude Index')

    #Saving to designated path on computer
    plot_path = os.path.join(folder_name, 'amplitudes_over_time_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def pressure_simulation(all_press_list, all_times_list, speeds, folder_name, n_thetas):

    """
    Function that creates a GIF animation of the pressure function over time.

    Inputs:
    all_press_list (list):  List of all pressure amplitudes at all time steps.
    all_times_list (list): List of all time steps.
    speeds (int): Speed of simulation (the higher, the faster).
    folder_name (string): Name of folder on the computer where the simulation should be saved to.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    gif_path (video/GIF): Saves video in the designated folder on your computer.
    
    """
    
    pressure_array = np.asanyarray(all_press_list) #Convert to array for easier handlings

    #Initialize Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim((-1, 1))
    ax.set_ylim((-0.1, 60))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pressure Amplitude')
    my_plot, = ax.plot([], [], '-')

    #Obtain sampling angles
    roots, weights = roots_legendre(500)
    float_roots = np.concatenate([np.flip(roots), roots])


    def drawframe(n): 
        if n < len(pressure_array): #Running for the duration of the simulation (until T_end)
            pressures = np.zeros(len(float_roots)) #Initialize pressure to 0
            for i in range(n_thetas-1):
                for ind_angles in range(len(float_roots)):
                    pressures[ind_angles] += pressure_array[n, i] * Legendre(np.cos(float_roots[ind_angles]), i, n_thetas)


            my_plot.set_data(float_roots, pressures*10)

        return my_plot,

    anim = animation.FuncAnimation(fig, drawframe, frames=len(pressure_array), interval=all_times_list[-1] / speeds, blit=True)

    return folder_name #Save to designated path on computer

def drop_simulation(all_amps_list, all_h_list, all_m_list, all_times_list, speeds, folder_name, n_thetas, view_vertical = 0, view_horizontal = 0):
    
    """
    Function that creates a GIF animation of the droplet over time.

    Inputs:
    all_amps_list (list): List of all amplitudes at all time steps.
    all_h_list (list): List of heights of the center of mass of the droplet at every time step.
    all_m_list (list): List of contact points at every time step.
    all_times_list (list): List of all time steps.
    speeds (int): Speed of simulation (the higher, the faster).
    folder_name (string): Name of folder on the computer where the simulation should be saved to.
    n_thetas (int): Total number of sampled angles.
    view_vertical (int): Rotation of the 3D plane in the vertical direction: 0 = straight view (not angled), positive values lead to view from the top.
    view_horizontal (int): Rotation of the 3D plane in the horizontal direction: 0 = straight/ front view (as in 2D), positive values rotate plane to the left (view from the right).

    Outputs:
    gif_path (video/GIF): Saves video in the designated folder on your computer.
    
    """

    #Define plotting angles (may be different from sample angles in the simulation, as you may want to use more angles for smoother visualization)
    roots, weights = roots_legendre(200)
    cos_theta_vec = np.concatenate(([1], np.flip(roots), [-1]))
    theta_vec = np.arccos(cos_theta_vec)
    thetas_3D = np.concatenate((np.array([-np.pi]),-np.flip(theta_vec[1:]), [0]))

    #Convert to array for easier handling
    all_amps_arr = np.asarray(all_amps_list)

    #Initialize plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.set_zlim((-0.5, 2.5))
    ax.set_xlabel('Height in R')
    ax.set_ylabel('Width in R')
    ax.view_init(view_vertical, view_horizontal)
    ax.set_box_aspect([1, 1, 1])  
    ax.set_title('Droplet Trajectory and Deformation')

    len_thetas = len(thetas_3D)
    layers = 100 #Layers of circles of the plots (more layers, more dense plotting)
    phi = np.linspace(0, (np.pi*2) + ((np.pi*2)/layers), layers) #Define phi for spherical plotting
    new_thetas = [theta + 3 * np.pi / 2 for theta in thetas_3D] #Define thetas in radians

    #Initialize plot data
    radii = np.ones(n_thetas) #Set undeformed radius
    for i in range(2, n_thetas): #Add deformation of every mode at every angle
        for ind_angles in range(n_thetas):
            radii[ind_angles] += all_amps_arr[0, i - 2] * Legendre(np.cos(thetas_3D[ind_angles]), i)


        #Transform to xy coordinates
        x = [radii[i] * np.cos(new_thetas[i]) for i in range(n_thetas)] 
        z = [radii[i] * np.sin(new_thetas[i]) + 1 - (1- all_h_list[0]) for i in range(n_thetas)]

        x_init = np.outer(x, np.sin(phi))
        y_init = np.outer(x, np.cos(phi)) #Create axisymmetry
        z_init = np.zeros_like(x_init)

        for i in range(len(x)):
            z_init[i:i+1,:] = np.full_like(z_init[0,:], z[i])

    #Add initial data to the plot
    my_plot = ax.plot_surface(x_init, y_init, z_init)

    def drawframe(n): 
        if n < len(all_amps_list): #Run function for remaining time (until T_end)
            radii = np.ones((len_thetas))
            for i in range(2, n_thetas):
                for ind_angles in range(len_thetas):
                    radii[ind_angles] += all_amps_list[n, i - 2] * Legendre(np.cos(thetas_3D[ind_angles]), i, n_thetas)


            x = [radii[i] * np.cos(new_thetas[i]) for i in range(len_thetas)]
            z = [radii[i] * np.sin(new_thetas[i]) + 1 - (1- all_h_list[n]) for i in range(len_thetas)]

            x_layer = np.outer(x, np.sin(phi))
            y_layer = np.outer(x, np.cos(phi))
            z_layer = np.zeros_like(x_layer)

            for i in range(len(x)):
                z_layer[i:i+1,:] = np.full_like(z_layer[0,:], z[i])

            ax.clear()
            ax.set_xlim((-1.5, 1.5))
            ax.set_ylim((-1.5, 1.5))
            ax.set_zlim((0, 3))
            ax.view_init(view_vertical, view_horizontal)
            ax.set_box_aspect([1, 1, 1])  
            my_plot = ax.plot_surface(x_layer, y_layer, z_layer, alpha=0.5)

            #Indicate whether in contact or not
            if all_m_list[n] != 0:
                my_plot = ax.scatter(1, 1, 2.5, color="red", s=100)
            else:
                my_plot = ax.scatter(1, 1, 2.5, color="green", s=100)


        return my_plot,

    anim = animation.FuncAnimation(fig, drawframe, frames=len(all_amps_list), interval=all_times_list[-1] / speeds, blit=True)
    return folder_name #Save to designated folder on device

def plot_and_save(rho_in_CGS, nu_in_GCS, sigma_in_CGS, g_in_CGS, R_in_CGS, V_in_CGS, T_end, Bond, Web, Ohn, n_thetas, n_sampling_time_L_mode):

    """
    Function that runs the simulation and saves all results and plots to a designated path on your computer.

    Inputs:
    n_thetas (int): Total number of sampled angles.
    n_sampling_time_L_mode (int):
    T_end (int): Duration of Simulation (in non-dimensional-time).

    either:
    rho_in_CGS (float): Density of chosen liquid in g/cm^3.
    sigma_in_CGS (float): Surface tension of chosen liquid in dyne/cm.
    g_in_CGS (float): Gravity in cm/s^2.
    R_in_CGS (float): Radius of droplet in cm.
    V_in_CGS (float): Velocity of the droplet's center of mass in cm/s.
    (and Bond & Web = None).

    or:
    Bond (float): Bond number, non-dimensional gravity.
    Web (float): Weber number, non-dimensional velocity.
    (and all others = None).

    Outputs: 
    Saves all plots and data to a designated path on your computer.d

    """

    #Dimensionless constansts
    if Bond == None:
        Bo = rho_in_CGS * g_in_CGS * R_in_CGS**2/ sigma_in_CGS
    else:
        Bo = Bond

    if Web == None:
        We = rho_in_CGS*V_in_CGS**2*R_in_CGS/sigma_in_CGS
    else:
        We = Web

    if Ohn == None:
        Oh = nu_in_GCS * np.sqrt((rho_in_CGS)/(sigma_in_CGS*R_in_CGS))
    else: 
        Oh = Ohn

    #Define paths to save data
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Brown', 'Thenarianto_comp_new') #Main directory
    folder_name = os.path.join(desktop_path, f'simulation_{R_in_CGS}_{V_in_CGS}_{round(Bo, 6)}_{round(We, 6)}') #Name of subdirectory

    #Create the main directory if it does not exist
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)

    #Create the subdirectory for the simulation
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    if Bond != None and Web != None: #If working with dimensionless constants
        V = -np.sqrt(We)
        H = 1 
    else: #If working with dimensional constants
        #Initial height
        H_in_CGS = R_in_CGS

        #units
        unit_length_in_CGS = R_in_CGS
        unit_time_in_CGS = np.sqrt(rho_in_CGS*R_in_CGS**3/sigma_in_CGS)
        unit_mass_in_CGS = rho_in_CGS*R_in_CGS**3

        #Dimensionless initial conditions
        V = -np.sqrt(We)
        H = H_in_CGS/unit_length_in_CGS


    #Running Simulations
    simulation_results = running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo, Oh)

    #Variables to store values of simulation
    all_amps_list = simulation_results[0]
    all_amps_vel_list = simulation_results[1]
    all_press_list = simulation_results[2]
    all_h_list = simulation_results[3]
    all_v_list = simulation_results[4]
    all_m_list = simulation_results[5]
    all_north_poles_list = simulation_results[6]
    all_south_poles_list = simulation_results[7]
    all_maximum_radii_list = simulation_results[8]
    all_times_list = simulation_results[9]

    #Reshaping lists to arrays for easier storing in CSV files
    amps_array = np.hstack(np.asarray(all_amps_list))
    all_amps_vel_array = np.hstack(np.asarray(all_amps_vel_list))
    all_press_array = (np.hstack(np.asarray(all_press_list)))
    all_h_array = (np.asarray(all_h_list)).reshape(len(all_h_list), 1)
    all_v_array = (np.asarray(all_v_list)).reshape(len(all_v_list), 1)
    all_m_array = (np.asarray(all_m_list)).reshape(len(all_m_list), 1)
    all_north_poles_array = (np.asarray(all_north_poles_list)).reshape(len(all_north_poles_list), 1)
    all_south_poles_array = (np.asarray(all_south_poles_list)).reshape(len(all_south_poles_list), 1)
    all_max_radii_array = (np.asarray(all_maximum_radii_list)).reshape(len(all_maximum_radii_list), 1)
    all_times_array = (np.asanyarray(all_times_list)).reshape(len(all_times_list), 1)


    #Calculating other variables
    coef_rest = coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_times_list, Bo, We)[0]
    contact_time = coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_times_list, Bo, We)[1]
    maxim_contact_radius = maximum_contact_radius(all_amps_list, all_m_list, n_thetas)
    print("Done with Coefficient of Restitution, Contact Time, Maximum Contact Radius") #To keep track of progress


    #Creating heights, radius and amplitudes plots and saving them
    height_plots(all_north_poles_list, all_south_poles_list, all_h_list, all_times_list, folder_name)
    print("Done with Height Plots")
    maximum_radius_plot(all_maximum_radii_list, all_times_list, folder_name)
    print("Done with Radius Plots")
    amplitudes_over_time_plot(all_amps_list, all_times_list, folder_name, n_thetas)
    print("Done with Amplitude Plots")

    tend = 100
    simulation_time = np.linspace(0, tend, tend + 1)
    speeds = 50

    #Creating video simulations
    drop_simulation(all_amps_list, all_h_list, simulation_time, speeds, os.path.join(folder_name, 'droplet_simulation.gif'), all_m_list, n_thetas, 0, 0, 0)
    pressure_simulation(all_press_list, simulation_time, speeds, 'pressure_simulation.gif', n_thetas, 0)

    #Save results to CSV file in main directory
    csv_file = os.path.join(desktop_path, 'simulation_results.csv')
    csv_header = ['R_in_CGS', 'V_in_CGS', 'Bo', 'We', 'Coefficient of Restitution', 'Contact Time', 'Maximum Radius']

    file_exists = os.path.isfile(csv_file)

    #Add to file if file exists in main directory
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(csv_header)
        writer.writerow([R_in_CGS, V_in_CGS, Bo, We, coef_rest, contact_time, maxim_contact_radius])


    #Convert data to arrays that can be stored easily and save as individual CSV files
    amps_file = os.path.join(folder_name, 'all_amps.csv')
    np.savetxt(amps_file, np.transpose(amps_array), delimiter=',')
    amps_vel_file = os.path.join(folder_name, 'all_amps_vel.csv')
    np.savetxt(amps_vel_file, np.transpose(all_amps_vel_array), delimiter=',')
    press_file = os.path.join(folder_name, 'all_press.csv')
    np.savetxt(press_file, np.transpose(all_press_array), delimiter=',')


    concatenated_array = np.hstack((all_times_array, all_h_array, all_v_array, all_m_array, all_north_poles_array, all_south_poles_array, all_max_radii_array))

    #Create CSV file in subdirectory for additional data 
    csv_file = os.path.join(folder_name, 'simulation_results_extra.csv')
    csv_header = ['Times', 'H', 'V', 'M', 'North Pole', 'South Pole', 'Maximum Radius']

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(csv_header)
        for row in concatenated_array:
            writer.writerow(row)


    return["All Files Saved"] #To keep track of progress