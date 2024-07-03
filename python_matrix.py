import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.special import roots_legendre
from scipy.special import eval_legendre
from scipy.special import legendre
from scipy.linalg import solve
from scipy.linalg import lstsq


#Physical constants
rho_in_SI = 1100
sigma_in_SI = 72E-3 #N/m
g_in_SI = 9.8 #m/s2

#Geometry
R_in_SI = 1E-3

#Initial speed in meters per second
V_in_SI = .0001

#Initial height
H_in_SI = R_in_SI

#units
unit_length = R_in_SI
unit_time = np.sqrt(rho_in_SI*R_in_SI**3/sigma_in_SI)
unit_mass = rho_in_SI*R_in_SI**3

#Total simulation time
T_end = 20 #in dimensionless units

#Numerical parameters
n_thetas = 40
n_poly = n_thetas
n_sampling_time_L_mode = 32 # don't use any less than 8


#Dimensionless constansts
We = rho_in_SI*V_in_SI**2*R_in_SI/sigma_in_SI
Bo = rho_in_SI * g_in_SI * R_in_SI/ sigma_in_SI
delta_time_max = (2*np.pi
                  /(np.sqrt(n_thetas*(n_thetas+2)*(n_thetas-1))
                    *n_sampling_time_L_mode))
n_times_base = int(np.ceil(T_end/delta_time_max))

#Initial conditions
V = -np.sqrt(We)
H = H_in_SI/unit_length

#define vector of cosines of the sampled angles
roots, weights = roots_legendre(n_thetas)
cos_theta_vec = np.concatenate(([1], np.flip(roots)))
theta_vec = np.arccos(cos_theta_vec)



def Legendre(x, n):  #define Legendre Polynomial of cosine
    """
    evaluates nth (int) Legendre polynomial at x (float)
    """
    if n == n_thetas and x != 1:
        value = 0
    else:
      value = eval_legendre(n, x)
    return value



np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def matrix(d_time, n_con): #define function for matrix

    """
    d_time (float): chosen time-step-size

    n_con (int): how many contact points are guesses

    """
    #First Block Row
    #Identity Matrix I(n_thetas -1)
    Identity_1 = np.eye((n_thetas-1)) #Condition Number = 1

    #Identity Matrix for (negative/ subtracting) delta_t
    Identity_delta_t = np.eye((n_thetas-1))*(-d_time) #Condition Number = 1

    #Zeros for the remaining first block-rock
    Zeros_1 = np.zeros(((n_thetas-1), (n_thetas+3)))

    #Concatenate the blocks horizontally
    block_row1 = np.concatenate((Identity_1, Identity_delta_t, Zeros_1), axis=1)


    #Second Block Row
    #changed but inaccurate
    C_list = [((l) * (l - 1) * (l + 2))*d_time for l in range(2, n_thetas+1)]
    C = np.diag(C_list)

    #Matrix for Pressure Amplitudes * time step
    D_1 = np.zeros((n_thetas-1, 2))

    D_list = [l*d_time for l in range(2, n_thetas+1)]
    D = np.diag(D_list)

    #FIlling Last two rows with zeros
    Zeros_2 = np.zeros((n_thetas-1, 2))

    #Concatenate sub-matrixes horizontally
    block_row2 = np.concatenate((C, Identity_1, D_1, D, Zeros_2), axis=1)


    #Third & Fourth (Contact & Non-Contact) Block Row
    E = np.zeros((n_thetas+1, 3 * n_thetas + 1))
    for i in range(n_con):
      cosin_value = cos_theta_vec[i]
      for j in range(2, n_thetas+1):
        E[i,j-2] = Legendre(cosin_value, j)

      #THIS IS F
      E[i,(n_thetas)*3-1] = (-1/(cosin_value))

    #THIS IS G
    for i in range(n_con, n_thetas+1):
      cosin_value = cos_theta_vec[i]
      for j in range(n_thetas+1):
        E[i, j + n_thetas * 2 -2] = Legendre(cosin_value, j)


    #Fourth Block Row #Condition Number of ~1
    last_identity = np.eye(2, (n_thetas*3+1), k=(n_thetas*3-1))
    last_identity[0, -1] = -d_time
    last_identity[1, (n_thetas*2-1)] = -d_time


    #Full Concatenation
    #block_full = np.concatenate((E, last_identity), axis=0)
    block_full = np.concatenate((block_row1, block_row2, E, last_identity), axis=0)

    return block_full

plt.spy((matrix(0.007, 5)))
plt.grid('on')



def error(amplitudes, cm_height, q): #Defining the Error function

    """
    inputs:

    amplitudes (list): list of amplitudes for q contact points at k+1

    cm_height (list): list of height of center of mass

    q (int): number of assumed contact points; (theoretical) maximum = n_thetas+1


    outputs:

    err (float): error at k+1, q
    """

    if q < 0:
        return np.infty

    if q == 0:
      all_heights = []
      for ind_theta in range(int(np.ceil((n_thetas/2)))): #only checks for thetas <= 90 deg
          val = cos_theta_vec[ind_theta]
          height = ((1 + np.sum([amplitudes[l]
                                * Legendre(val, l+2) for l in range(n_thetas-1)]))
                        * val)
          all_heights.append(height)

      for height in all_heights:
        if height > cm_height:
          return np.infty
      return np.infty

    #variables to store sums
    sum_q = 1
    sum_qp1 = 1

    #extracting the cos(theta) values at q (= q-1 in list index) and q+1 (= q in list index)
    q_val = cos_theta_vec[q-1]
    qp1_val = cos_theta_vec[q]

    for l in range(n_thetas-1):
      sum_q += amplitudes[l] * Legendre(q_val,l+2)
      sum_qp1 += amplitudes[l] * Legendre(qp1_val,l+2)

    #calculating the vertical components of each point
    vertical_q = sum_q * q_val
    vertical_qp1 = sum_qp1 * qp1_val

    err1 = np.abs(vertical_q - vertical_qp1)
    err2 = 0
    for ind_theta in range(q,n_thetas+1):
      rad = 1
      for ind_poly in range(2,n_thetas+1):
        rad += amplitudes[ind_poly-2] * Legendre(cos_theta_vec[ind_theta],ind_poly)
      if rad*cos_theta_vec[ind_theta]>cm_height:        
        err2 = np.infty
        break

           
    err = np.max([err1,err2])
    print("Error 1:", err1, "Error 2:", err2)

    return err



def try_q(amplitudes, amplitude_velocities, height_cm, velocity_cm, q, delta_t):

  """
  inputs:

  amplitudes (list): list of amplitudes at the previous time (k)

  amplitude_velocities (list): list of velocities of amplitudes at the previous time (k)

  height_cm (float): height of the center of mass at the previous time (k)

  height_cm (float): velocity of the center of mass at the previous time (k)

  q (int): number of contact points to be tried at k+1 (theoretical) maximum = n_thetas+1

  delta_t (float): time_step (in non-dimensional time)


  outputs:

  matrix_solution (list): list of the solution of the system of equations at q contact points
                          at time k+1

  err (float): error at k+1, q

  """
  #adjust velocity of center of mass
  edited_velocity = velocity_cm - (Bo * delta_t)
  print("q = ", q)


  if q < 0:
    matrix_solution = amplitudes + amplitude_velocities + np.zeros(n_thetas + 1).tolist() + [height_cm] + [edited_velocity]
    err = error(amplitudes, height_cm, q)

  elif q == 0:
    #extracting modes
    modes_list = amplitudes + amplitude_velocities
    #extracting height & velocity of center of mass
    cm_list = [height_cm] + [edited_velocity]

    #extracting parts of the matrices for modes and cm
    modes_matrix = matrix(delta_t, q)[: int(n_thetas*2-2), : int(n_thetas*2-2)]
    cm_matrix = matrix(delta_t, q)[-2:,-2:]

    #solving systems of equations
    solution_modes = solve(modes_matrix, modes_list)
    solutions_cm = solve(cm_matrix, cm_list)
    matrix_solution = np.concatenate((solution_modes, np.zeros(n_thetas+1), solutions_cm),0).tolist()

    #calculating the error at k+1, q
    err = error(solution_modes[: n_thetas-1], solutions_cm[0], q)

  else:
    b_list = amplitudes + amplitude_velocities + [-1 for i in range(q)] + np.zeros((n_thetas+1-q)).tolist() + [height_cm] + [edited_velocity]
    full_matrix = (matrix(delta_t, q))
    matrix_solution = (solve(full_matrix, b_list)).tolist()

    #calculating the error at k+1, q
    err = error(matrix_solution[: n_thetas-1], matrix_solution[-2], q)
    amplitudes = matrix_solution[: n_thetas-1]
    amplitude_velocities = matrix_solution[n_thetas-1: (2*n_thetas-2)]
    height_cm = matrix_solution[-2]
    velocity_cm = matrix_solution[-1]
    

  return [amplitudes, amplitude_velocities, height_cm, velocity_cm, matrix_solution[2*n_thetas-2: -2], err]



#initialize list of times
times_list = (np.linspace(0,n_times_base,n_times_base+1)
              * delta_time_max).tolist()



#main time loop
current_amplitudes = np.zeros(n_thetas-1).tolist()
current_ampl_velocities = np.zeros(n_thetas-1).tolist()
current_cm_height = H
current_cm_velocity = V
current_pressure = 0

#Variables to store values of simulation
current_q = 0
all_amplitudes = [[current_amplitudes]]
all_pressures = []
all_heigts = [current_cm_height]
ind_time = 0 #this counts the last time for which we have a successful solution

while times_list[ind_time] < T_end:

    #Time step is the difference in the times_list
    delta_t = times_list[ind_time+1] - times_list[ind_time]

    #print("🏳️ At the beginning of the loop")
    print("⏰ Time Index:", ind_time)
    #print("🕰️ Time Step:", delta_t)
    print("🏴 Current q:", current_q)
    print("📏 Current Height ", current_cm_height)
    #print("Amplitudes ",current_amplitudes)

    #Set all errors to infinity
    try_q_contact, try_qm1_contact, try_qm2_contact, try_qp1_contact, try_qp2_contact = np.infty, np.infty, np.infty, np.infty, np.infty

    #calculate errors for initial comparison
    try_q_contact = try_q(current_amplitudes, current_ampl_velocities, current_cm_height, current_cm_velocity, current_q, delta_t)
    print("Error q:", try_q_contact[-1])
    try_qm1_contact = try_q(current_amplitudes, current_ampl_velocities, current_cm_height, current_cm_velocity, (current_q-1), delta_t)
    print("Error -1:", try_qm1_contact[-1])

    #if one contact point less is better than the previous contact points, test two contact points less
    if try_qm1_contact[-1] < try_q_contact[-1]:
        print("✔️ 1 smaller better")
        try_qm2_contact = try_q(current_amplitudes, current_ampl_velocities, current_cm_height, current_cm_velocity, (current_q-2), delta_t)
        if try_qm2_contact[-1] < try_qm1_contact[-1]:
          print("Error q -2", try_qm2_contact[-1])
          #print("✋ 2 smaller best")

          #insert new time step
          time_insert = (times_list[ind_time] + times_list[ind_time+1])/2
          times_list.insert(ind_time+1, time_insert)

        else:
          #print("✅ 1 smaller best")
          ind_time += 1
          current_q = current_q - 1
          current_amplitudes = try_qm1_contact[0]
          current_ampl_velocities = try_qm1_contact[1]
          current_cm_height = try_qm1_contact[2]
          current_cm_velocity = try_qm1_contact[3]
          current_pressure = try_qm1_contact[4]
          all_pressures.append(current_pressure)
          all_amplitudes.append(current_amplitudes)
          all_heigts.append(current_cm_height)

    else:
      #print("❌ 1 smaller not better")
      try_qp1_contact = try_q(current_amplitudes, current_amplitudes, current_cm_height, current_cm_velocity, (current_q+1), delta_t)
      print("Error +1:", try_qp1_contact[-1])

      if try_qp1_contact[-1] < try_q_contact[-1]:
       # print("✔️ 1 greater better")
        try_qp2_contact = try_q(current_amplitudes, current_amplitudes, current_cm_height, current_cm_velocity, (current_q+2), delta_t)
        if try_qp2_contact[-1] < try_qp1_contact[-1]:
          #print("✋ 2 greater best")
          print("Error q +2", try_qp2_contact[-1])
          #insert new time step
          time_insert = (times_list[ind_time] + times_list[ind_time+1])/2
          times_list.insert(ind_time+1, time_insert)

        else:
          #accept solution and move forward in time
          #print("✅ 1 greater best")
          ind_time += 1
          current_q += 1
          current_amplitudes = try_qp1_contact[0]
          current_ampl_velocities = try_qp1_contact[1]
          current_cm_height = try_qp1_contact[2]
          current_cm_velocity = try_qp1_contact[3]
          current_pressure = try_qp1_contact[4]
          all_amplitudes.append(current_amplitudes)
          all_heigts.append(current_cm_height)
          all_pressures.append(current_pressure)

      else: #Stick to the same q and move forward in time.
        ind_time += 1
        current_q += 0
        current_amplitudes = try_q_contact[0]
        current_ampl_velocities = try_q_contact[1]
        current_cm_height = try_q_contact[2]
        current_cm_velocity = try_q_contact[3]
        current_pressure = try_q_contact[4]
        all_pressures.append(current_pressure)
        all_amplitudes.append(current_amplitudes)
        all_heigts.append(current_cm_height)
        #print("✅ current best")





