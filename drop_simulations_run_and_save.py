"""
Importing necessary packages
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scipy.special import eval_legendre
from scipy.linalg import solve
from matplotlib import animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import csv


def Legendre(x, n, n_thetas):  #Define Legendre Polynomial of cosine
    """
    Function that evaluates nth Legendre polynomial at x with a scipy function eval_legendre.

    Inputs:
    x (float): X-value at which Legendre polynomial is evaluated
    n (int): Number of Legendre polynomial to be evaluated

    Output:
    value (float): value of nth Legendre polynomial at x

    """
    if n == n_thetas and x != 1:
        #To avoid numerical errors very close to zero
        value = 0
    else:
      value = eval_legendre(n, x)
    return value

def matrix(delta_t,q,n_thetas,cos_theta_vec,Oh): #define function for matrix

    """
    Function that creates the matrix with a given time step and a given number of conact points.
    
    Inputs:
    delta_t (float): Chosen time-step-size.
    q (int): Number of guessed contact points.

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
        FH_mat[ind_angle,ind_poly] = Legendre(cos_theta_vec[ind_angle],ind_poly,n_thetas)
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
    #sixth row of blocks
    K = np.zeros((1,n_thetas+1))
    K[0,1] = -1
    sixth_block_row = np.concatenate((np.zeros((1,n_thetas-1)),
                                      np.zeros((1,n_thetas-1)),
                                      delta_t*K,
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
                *Legendre(cos_theta_vec[ind_theta],ind_poly,n_thetas)
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
    #Right-hand side for cm equations
    RHS_h_vec = np.concatenate((h_prev,
                                v_prev-(delta_t*Bo)
                               ),
                               0
                              )

    #Extracting parts of the matrices for modes and cm
    M_modes_mat = matrix(delta_t, q, n_thetas,cos_theta_vec,Oh)[: int(n_thetas*2-2), : int(n_thetas*2-2)]
    M_h_mat = matrix(delta_t, q, n_thetas,cos_theta_vec,Oh)[-2:,-2:]

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
    M_mat = (matrix(delta_t, q, n_thetas, cos_theta_vec,Oh))
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

def height_poles(all_amps_vec, h, n_thetas):

    """
    Function that evaluates the deformed droplet's north and south pole radii.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at a single point in time.
    h (float): Height of the center of mass.
    n_thetas (int): Total number of sampled angles.

    Outputs:
    north_pole_radius (vector): Radius at the north pole.
    south_pole_radius (vector): Radius at the south pole.

    """

    #Setting undeformed radii
    north_pole_radius = 1
    south_pole_radius = 1

    #Adding deformations from the Legendre modes
    for i in range(2, n_thetas):
        north_pole_radius += all_amps_vec[i-2] * Legendre(-1, i, n_thetas)
        south_pole_radius += all_amps_vec[i-2] * Legendre(1, i, n_thetas)

    #Obtaining height
    north_pole_height = north_pole_radius + h
    south_pole_height = h - south_pole_radius

    return[north_pole_height, south_pole_height]

def min_north_pole_height(all_north_poles_list, all_times_list):
    min_north_pole = np.min(all_north_poles_list)
    min_north_pole_ind = all_north_poles_list.index(min_north_pole)
    time_of_min_north_pole = all_times_list[min_north_pole_ind]
    return [min_north_pole, time_of_min_north_pole]

def max_radius_at_each_t(all_amps_vec, n_thetas, theta_vec):

    """
    Function that calculates the maximum radius of the droplet at a single point in time.

    Inputs:
    amps_prev_vec (vector): Vector of amplitudes at a single point in time.

    Outputs:
    radius_of_largest_deformation (vector): Radius of the angle, where largest radius is achieved.

    """

    radii = np.ones(n_thetas) #Setting radi at all sampled points to an undeformed radius
    for i in range(2, n_thetas):
        for ind_angles in range(n_thetas): #Adding deformation
            radii[ind_angles] += all_amps_vec[i - 2] * Legendre(np.cos(theta_vec[ind_angles]), i, n_thetas)

    radius_of_largest_deformation = np.max(radii)
    return radius_of_largest_deformation #Returning largest radius

def max_radius_over_t(all_max_radii, contact_time_index, all_times_list):
    relevant_max_radii = all_max_radii[: contact_time_index]
    max_radius = np.max(relevant_max_radii)
    max_radius_index = relevant_max_radii.index(max_radius)
    time_of_max_radius = all_times_list[max_radius_index]

    return [max_radius, time_of_max_radius]

def alpha_coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_time_list, Bo, We): 

    """
    Function that calculates the coefficient of restitution and contact time for a droplet impact.

    Inputs:
    all_h_list (list): List of heights of the center of mass of the droplet at every time step.
    all_v_list (list): List of velocities of the center of mass of the droplet at every time step.
    all_m_list (list): List of contact points at every time step.
    all_time_list (list): List of all time steps.

    Outputs:
    coeff_rest_squared_float (float): Coefficient of Restitution.
    contact_time_float (float): Time of contact (non-dimensional).

    """

    #Setting all values to be obtained to None to avoid errors
    final_velocity = None
    final_height = None
    contact_time = None
    index = None

    for i in range(1, len(all_m_list)-1):
        if all_m_list[i] == 0 and all_m_list[i-1] > 0: #Marks end of contact
            index = i
            break

    #Handling if no take-off occurs
    if index == None:
        return [None, None, None]

    #Averages of last contact and first non-comtact times
    final_velocity = (all_v_list[index] + all_v_list[index-1])/2
    final_height = (all_h_list[index] + all_h_list[index-1])/2
    contact_time = (all_time_list[index] + all_time_list[index-1])/2
    last_time_in_contact =  index


    #Calculating of coefficient of restitution squared
    coeff_rest_squared = (((Bo * (final_height - 1))
                          + 0.5 * (final_velocity**2))/
                          (0.5 * (We)))

    coeff_rest_squared_float = np.sqrt(np.abs(float(coeff_rest_squared)))


    return [coeff_rest_squared_float, contact_time, last_time_in_contact]

def center_coefficient_of_restitution(all_v_list, all_m_list):
    
    final_velocity = None
    initial_velocity = None
    index = None

    for ind in range(len(all_m_list)):
        if all_m_list[ind] == 0:
            if all_m_list[ind+1] != 0:
                initial_velocity = (all_v_list[ind]+all_v_list[ind+1])/2
                break

    for i in range(1, len(all_m_list)-1):
        if all_m_list[i] == 0 and all_m_list[i-1] > 0: #Marks end of contact
            index = i
            break
    #Handling if no take-off occurs
    if index == None:
        return None

    #Averages of last contact and first non-comtact times
    final_velocity = (all_v_list[index] + all_v_list[index-1])/2
    coef_rest = (-final_velocity)/(initial_velocity)
    return float(coef_rest)

def maximum_contact_radius(all_amps_list, all_m_list, all_times_list, n_thetas, theta_vec):

    """
    Function that calculates the maximum contact radius of a droplet impact.

    Inputs:
    all_amps_list (list): List of all amplitudes at every time step.
    all_m_list (list): List of contact points at every time step.

    Outputs:
    np.max(contact_radii) (float): Maximum contact radius achieved during impact.

    """

    contact_radii = []
    all_amps_mat = np.asarray(all_amps_list) #Converting to array for easier handling
    for time in range(len(all_amps_list)):
        m = all_m_list[time]
        radii_vec = np.ones(2)
        if m == 0: #If not in contact, contact area = 0
            contact_radii.append(0)
        else:
            for i in range(2, n_thetas):
                radii_vec[0] += all_amps_mat[time, i - 2] * Legendre(np.cos(theta_vec[m-1]), i, n_thetas)
                radii_vec[1] += all_amps_mat[time, i - 2] * Legendre(np.cos(theta_vec[m]), i, n_thetas)

            x_1 = radii_vec[0] * np.sin(theta_vec[m-1])
            x_2 = radii_vec[1] * np.sin(theta_vec[m])

            #Calculating average of last contact-point-radius and first non-contact-point radius
            contact_radii.append((x_1+x_2)/2)

    max_contact_radius = np.max(contact_radii)
    maximum_contact_radius_ind = contact_radii.index(max_contact_radius)
    time_of_mas_radius = all_times_list[maximum_contact_radius_ind]

    return [max_contact_radius, time_of_mas_radius] #Returning largest contact area

def max_radial_projection(all_amps_list, n_thetas_int, all_times_list, theta_vec):

    all_amps_array = np.asarray(all_amps_list)
    max_horizontal_projection = []

    for time_ind in range(len(all_amps_array)):
        all_horiz_comps = []
        thetas = np.concatenate((theta_vec,np.array([np.pi])), 0)
        radii = np.ones(len(thetas))

        for ind_amplitude in range(2, n_thetas_int): #Add deformation of every mode at every angle
            for ind_angles in range(n_thetas_int):
                radii[ind_angles] += all_amps_array[time_ind, ind_amplitude - 2] * Legendre(np.cos(thetas[ind_angles]), ind_amplitude, n_thetas_int)

        for i in range(len(thetas)):
            horizontal_component = (radii[i] * np.sin(thetas[i]))
            all_horiz_comps.append(horizontal_component)

        max_horizontal_projection.append(np.max(all_horiz_comps))


    max_of_radial_projections = np.max(max_horizontal_projection)
    max_radial_projection_index = max_horizontal_projection.index(max_of_radial_projections)
    time_of_max_view = all_times_list[max_radial_projection_index]

    return [max_of_radial_projections, time_of_max_view]

def min_side_height(all_amps_list, n_thetas_int, all_times_list, all_h_list, theta_vec):


    all_amps_array = np.asarray(all_amps_list)
    highest_side_view = []

    for time_ind in range(len(all_amps_array)):
        all_heights = []
        thetas = np.concatenate((theta_vec,np.array([np.pi])), 0)
        radii = np.ones(len(thetas))

        for ind_amplitude in range(2, n_thetas_int): #Add deformation of every mode at every angle
            for ind_angles in range(n_thetas_int):
                radii[ind_angles] += all_amps_array[time_ind, ind_amplitude - 2] * Legendre(np.cos(thetas[ind_angles]), ind_amplitude, n_thetas_int)

        for i in range(len(thetas)):
            y = all_h_list[time_ind] - (radii[i] * np.cos(thetas[i]))
            all_heights.append(y)

        highest_side_view.append(np.max(all_heights))


    min_of_highest_side_view = np.min(highest_side_view)
    min_side_view_index = highest_side_view.index(min_of_highest_side_view)
    time_of_min_view = all_times_list[min_side_view_index]

    return [min_of_highest_side_view, time_of_min_view]

def running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo, theta_vec, Oh):

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
    all_north_poles_list = [np.asarray([height_poles(amps_prev_vec, H, n_thetas)[0]])]
    all_south_poles_list = [np.asarray([height_poles(amps_prev_vec, H, n_thetas)[1]])]
    all_maximum_radii_list = [max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec)] 

    #to stop after lift-off
    lift_off = False
    count = 0
    
    
    while all_times_list[ind_time] < all_times_list[-1]:
        #print(m_prev)

        if len(all_m_list) > 2:
            if all_m_list[-1] == 0:
                if all_m_list[-2] > 0:
                    lift_off = True
        if lift_off == True:
            count += 1
        if count == 50:
            break


        #Time step is the difference in the times_list
        delta_t = all_times_list[ind_time+1] - all_times_list[ind_time]

        #Set all errors to infinity
        err_m_prev = np.inf
        err_m_prev_m1 = np.inf
        err_m_prev_m2 = np.inf
        err_m_prev_p1 = np.inf
        err_m_prev_p2 = np.inf

        #calculate errors for initial comparison
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

        #if one contact point less is better than the previous contact points, test two contact points less
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

                #insert new time step
                time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1])/2
                all_times_list.insert(ind_time+1, time_insert)

            else:
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
                all_north_poles_list.append(height_poles(amps_prev_vec, h_prev, n_thetas)[0])
                all_south_poles_list.append(height_poles(amps_prev_vec, h_prev, n_thetas)[1])
                all_maximum_radii_list.append(max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec))

        else:
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
                if err_m_prev_p2 < err_m_prev_p1:
                #insert new time step
                    time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1])/2
                    all_times_list.insert(ind_time+1, time_insert)
                else:
                    #accept solution and move forward in time
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
                    all_north_poles_list.append(height_poles(amps_prev_vec, h_prev, n_thetas)[0])
                    all_south_poles_list.append(height_poles(amps_prev_vec, h_prev, n_thetas)[1])
                    all_maximum_radii_list.append(max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec))
                
            elif err_m_prev_p1 == np.inf and err_m_prev == np.inf and err_m_prev_m1 == np.inf:
                #insert new time step
                time_insert = (all_times_list[ind_time] + all_times_list[ind_time+1])/2
                all_times_list.insert(ind_time+1, time_insert)

            else: #Stick to the same q and move forward in time.
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
                all_north_poles_list.append(height_poles(amps_prev_vec, h_prev, n_thetas)[0])
                all_south_poles_list.append(height_poles(amps_prev_vec, h_prev, n_thetas)[1])
                all_maximum_radii_list.append(max_radius_at_each_t(amps_prev_vec, n_thetas, theta_vec))

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

    all_north_poles_float_list = [arr.item() for arr in all_north_poles_list]
    all_south_poles_float_list = [arr.item() for arr in all_south_poles_list]
    all_h_float_list = [arr.item() for arr in all_h_list]

    plt.figure()
    plt.plot(all_times_list, all_north_poles_float_list, label="North Pole", color="blue")
    plt.plot(all_times_list, all_south_poles_float_list, label="South Pole", color="red")
    plt.plot(all_times_list, all_h_float_list, label="Center of Mass", color="green")
    plt.legend()
    plt.xlabel("Non-Dimensional Time")
    plt.ylabel("North/ South Pole/ Center of Mass in Radii")
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

    plt.figure()
    plt.plot(all_times_list, all_maximum_radii_list)
    plt.ylabel("Maxmum Radius in Non-Dimensional Radii")
    plt.xlabel("Non-Dimensional Time")
    plt.legend()
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

    amplitudes_array = np.asarray(all_amps_list)

    fig, ax = plt.subplots()
    colors = cm.viridis(np.linspace(0, 1, n_thetas))

    for amp_ind in range(n_thetas-1):
        coefficients_list = []
        for time_ind in range(len(amplitudes_array)):
            coefficients_list.append(amplitudes_array[time_ind, amp_ind])
        ax.plot(all_times_list, coefficients_list, color=colors[amp_ind])
    
    plt.ylabel("Amplitude")
    plt.xlabel("Non-Dimensional Time")

     # Create a ScalarMappable and add a colorbar
    norm = mcolors.Normalize(vmin=2, vmax=n_thetas)
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])  # Only needed for older versions of Matplotlib

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([2, n_thetas])
    cbar.set_ticklabels([2, n_thetas])
    cbar.set_label('Amplitude Index')

    plot_path = os.path.join(folder_name, 'amplitudes_over_time_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_and_save(rho_in_CGS, sigma_in_CGS, nu_in_GCS, g_in_CGS, R_in_CGS, V_in_CGS, T_end, Bond, Web, Ohn, n_thetas, n_sampling_time_L_mode):

    """
    Function that runs the simulation and saves all results and plots to a designated path on your computer.

    Inputs:
    n_thetas (int): Total number of sampled angles.
    n_sampling_time_L_mode (int):
    T_end (int): Duration of Simulation (in non-dimensional-time).

    either:
    rho_in_CGS (float): Density of chosen liquid in g/cm^3.
    sigma_in_CGS (float): Surface tension of chosen liquid in dyne/cm.
    nu_in_GCS (float): in cm^2/s.
    g_in_CGS (float): Gravity in cm/s^2.
    R_in_CGS (float): Radius of droplet in cm.
    V_in_CGS (float): Velocity of the droplet's center of mass in cm/s.
    (and Bond & Web = None).

    or:
    Bond (float): Bond number, non-dimensional gravity. If you want to define from R_in_CGS, sigma_in_CGS, rho_in_CGS, and g_in_CGS: input: None.
        If want to set it to 0: Input 0
    Web (float): Weber number, non-dimensional velocity. If you want to define from R_in_CGS, sigma_in_CGS, rho_in_CGS, and V_in_CGS: input: None.
        If want to set it to 0: Input 0   
    Ohn (float): Ohnesorghe number, non-dimensional viscosity. If you want to define from R_in_CGS, sigma_in_CGS, rho_in_CGS, and nu_in_CGS: input: None.
        If want to set it to 0: Input 0
    (and all others = None).

    Outputs: 
    Saves all plots and data to a designated path on your computer.d

    """

    roots, weights = roots_legendre(n_thetas)
    cos_theta_vec = np.concatenate(([1], np.flip(roots)))
    theta_vec = np.arccos(cos_theta_vec)
    q_last = np.argmax(cos_theta_vec <= 0)

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

    print(We, Bo, Oh)
    desktop_path = os.path.join('Desktop', 'Try')
    folder_name = os.path.join(desktop_path, f'simulation_{R_in_CGS}_{V_in_CGS}_{round(Bo, 5)}_{round(We, 5)}_{round(Oh, 5)}')

    # Create the main directory if it does not exist
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)

    # Create the subdirectory for this simulation
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #units
    unit_length_in_CGS = R_in_CGS
    unit_time_in_CGS = np.sqrt(rho_in_CGS*R_in_CGS**3/sigma_in_CGS)
    unit_mass_in_CGS = rho_in_CGS*R_in_CGS**3

    if Bond == None and Web == None:
        V = -np.sqrt(We)
        H = 1 #H_in_CGS/unit_length_in_CGS

    else:
        #Initial height
        H_in_CGS = R_in_CGS

        #Dimensionless initial conditions
        V = -np.sqrt(We)
        H = H_in_CGS/unit_length_in_CGS


    #Running Simulations
    simulation_results = running_simulation(n_thetas, n_sampling_time_L_mode, T_end, H, V, Bo,theta_vec, Oh)
    print("Done with Simulation")

    #Variables to store values of simulation
    all_amps_list = simulation_results[0]
    all_amps_vel_list = simulation_results[1]
    all_press_list = simulation_results[2]
    all_h_list = simulation_results[3]
    all_v_list = simulation_results[4]
    all_m_list = simulation_results[5]
    all_north_poles_list = simulation_results[6]
    #all_south_poles_list = simulation_results[7]
    all_maximum_radii_list = simulation_results[8]
    all_times_list_original = simulation_results[9]
    all_times_list = all_times_list_original[: len(all_m_list)]


    #Reshaping lists to arrays for easier storing in CSV files
    all_amps_array = np.hstack(np.asarray(all_amps_list))
    all_amps_vel_array = np.hstack(np.asarray(all_amps_vel_list))
    all_press_array = (np.hstack(np.asarray(all_press_list)))
    all_h_array = (np.asarray(all_h_list)).reshape(len(all_h_list), 1)
    all_v_array = (np.asarray(all_v_list)).reshape(len(all_v_list), 1)
    all_m_array = (np.asarray(all_m_list)).reshape(len(all_m_list), 1)
    # all_north_poles_array = (np.asarray(all_north_poles_list)).reshape(len(all_north_poles_list), 1)
    # all_south_poles_array = (np.asarray(all_south_poles_list)).reshape(len(all_south_poles_list), 1)
    # all_max_radii_array = (np.asarray(all_maximum_radii_list)).reshape(len(all_maximum_radii_list), 1)
    all_times_array = (np.asanyarray(all_times_list)).reshape(len(all_times_list), 1)


    #Calculating other variables
    center_coef_rest = center_coefficient_of_restitution(all_v_list, all_m_list)
    alpha_coef_rest = alpha_coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_times_list, Bo, We)[0]
    contact_time_nd = alpha_coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_times_list, Bo, We)[1]
    print("coef rests:", center_coef_rest, alpha_coef_rest)
    contact_time_ind = alpha_coefficient_of_restitution(all_h_list, all_v_list, all_m_list, all_times_list, Bo, We)[2]
    min_north_pole_h_nd = min_north_pole_height(all_north_poles_list, all_times_list)[0]
    time_min_north_pole_h_nd = min_north_pole_height(all_north_poles_list, all_times_list)[1]
    max_radius_nd = max_radius_over_t(all_maximum_radii_list, contact_time_ind, all_times_list)[0]
    time_max_radius_nd = max_radius_over_t(all_maximum_radii_list, contact_time_ind, all_times_list)[1]
    max_contact_radius_nd = maximum_contact_radius(all_amps_list, all_m_list, all_times_list, n_thetas, theta_vec)[0]
    time_max_contact_radius_nd = maximum_contact_radius(all_amps_list, all_m_list, all_times_list, n_thetas, theta_vec)[1]
    max_radial_project_nd = max_radial_projection(all_amps_list, n_thetas, all_times_list, theta_vec)[0]
    time_max_radial_project_nd = max_radial_projection(all_amps_list, n_thetas, all_times_list, theta_vec)[1]
    min_side_h_nd = min_side_height(all_amps_list, n_thetas, all_times_list, all_h_list, theta_vec)[0]
    time_min_side_h_nd = min_side_height(all_amps_list, n_thetas, all_times_list, all_h_list, theta_vec)[1]
    print("Done with Coefficient of Restitution, Contact Time, Maximum Contact Radius")

    if contact_time_nd is not None:
        contact_time_cgs = contact_time_nd*unit_time_in_CGS
        print(contact_time_cgs)
    else:
        contact_time_cgs = None
    min_north_pole_h_cgs = min_north_pole_h_nd*R_in_CGS
    time_min_north_pole_h_cgs = time_min_north_pole_h_nd*unit_time_in_CGS
    max_radius_cgs = max_radius_nd*R_in_CGS
    time_max_radius_cgs = time_max_radius_nd*unit_time_in_CGS
    max_contact_radius_cgs = max_contact_radius_nd*R_in_CGS
    time_max_contact_radius_cgs = time_max_contact_radius_nd*unit_time_in_CGS
    max_radial_project_cgs = max_radial_project_nd*R_in_CGS
    time_max_radial_project_cgs = time_max_radial_project_nd*unit_time_in_CGS
    min_side_h_cgs = min_side_h_nd*R_in_CGS
    time_min_side_h_cgs = time_min_side_h_nd*unit_time_in_CGS



    # #Creating heights, radius and amplitudes plots and saving them
    # height_plots(all_north_poles_list, all_south_poles_list, all_h_list, all_times_list, folder_name)
    # maximum_radius_plot(all_maximum_radii_list, all_times_list, folder_name)
    # amplitudes_over_time_plot(all_amps_list, all_times_list, folder_name, n_thetas)
    # print("Done with  Plots")

    #tend = 100
    #simulation_time = np.linspace(0, tend, tend + 1)
    #speeds = 50
    #drop_simulation(all_amps_list, all_h_list, simulation_time, speeds, os.path.join(folder_name, 'droplet_simulation.gif'), all_m_list, n_thetas, 0, 0, 0)
    #pressure_simulation(all_press_list, simulation_time, speeds, 'pressure_simulation.gif', n_thetas, 0)
    # Save results to CSV
    csv_file = os.path.join(desktop_path, 'simulation_results.csv')
    csv_header = ['R_in_CGS', 'V_in_CGS', 
                  'Bo', 'We', 'Oh',
                  "Alpha Coefficient of restitution", "Center Coefficient of restitution", 
                  'Contact time ND', 'Contact time CGS',
                  'Min north pole height ND', 'Min north pole height CGS',
                  'Time min north pole height ND', 'Time min north pole height CGS',
                  'Max radius ND', 'Max radius CGS',
                  'Time of max radius ND', 'Time of max radius CGS',
                  'Max contact radius ND', 'Max contact radius CGS',
                  'Time max contact radius ND', 'Time max contact radius CGS',
                  'Max radial projection ND', 'Max radial projection CGS',
                  'Time max radial projection ND', 'Time max radial projection CGS',
                  'Min side height ND', 'Min side height CGS',
                  'Time min side height ND', 'Time min side height CGS' 
                  ]

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(csv_header)
        writer.writerow([R_in_CGS, V_in_CGS,
                        Bo, We, Oh, 
                        alpha_coef_rest, center_coef_rest,
                        contact_time_nd, contact_time_cgs,
                        min_north_pole_h_nd, min_north_pole_h_cgs,
                        time_min_north_pole_h_nd, time_min_north_pole_h_cgs,
                        max_radius_nd, max_radius_cgs,
                        time_max_radius_nd, time_max_radius_cgs,
                        max_contact_radius_nd, max_contact_radius_cgs,
                        time_max_contact_radius_nd, time_max_contact_radius_cgs,
                        max_radial_project_nd, max_radial_project_cgs,
                        time_max_radial_project_nd, time_max_radial_project_cgs,
                        min_side_h_nd, min_side_h_cgs,
                        time_min_side_h_nd, time_min_side_h_cgs])


    amps_file = os.path.join(folder_name, 'all_amps.csv')
    np.savetxt(amps_file, np.transpose(all_amps_array), delimiter=',')
    amps_vel_file = os.path.join(folder_name, 'all_amps_vel.csv')
    np.savetxt(amps_vel_file, np.transpose(all_amps_vel_array), delimiter=',')
    press_file = os.path.join(folder_name, 'all_press.csv')
    np.savetxt(press_file, np.transpose(all_press_array), delimiter=',')


    concatenated_array = np.hstack((all_times_array, all_h_array, all_v_array, all_m_array)) #, all_north_poles_array, all_south_poles_array, all_max_radii_array))

    csv_file = os.path.join(folder_name, 'simulation_results_extra.csv')
    csv_header = ['Times', 'H', 'V', 'M'] #, 'North Pole', 'South Pole', 'Maximum Radius at t']

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(csv_header)
        for row in concatenated_array:
            writer.writerow(row)


    return["All Files Saved"]
