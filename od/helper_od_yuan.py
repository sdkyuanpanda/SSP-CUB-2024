"""
Helper funtions for OD lib code.
"""

import numpy as np
from astroquery.jplhorizons import Horizons

def vel_arr_to_gaussian(arr):
    CONVERSION_FACTOR = 365.2568983 / (2*np.pi)
    arr *= CONVERSION_FACTOR
    return arr

def get_sun_earth_vector(time):
    sun = Horizons(id = "@sun", location = "500")
    sun.epochs = time
    vecs = sun.vectors(refplane = "earth")
    return np.array([float(vecs["x"]), float(vecs["y"]), float(vecs["z"])]).astype(np.float64)

def parse_custom_csv_input(filename):
    times = []
    RA_vec = []
    DEC_vec = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            parts = line.split(",")
            times.append(float(parts[0]))
            RA_vec.append(ra_to_deg(int(parts[1]), int(parts[2]), float(parts[3]), True))
            DEC_vec.append(dec_dms_to_deg(int(parts[4]), int(parts[5]), float(parts[6])) * np.pi/180)
    return np.array(times), np.array(RA_vec), np.array(DEC_vec)

def parse_horizon_vec_table_txt(file_path):
    time = []
    position = []
    velocity = []
    parse = False
    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if "$$SOE" in line:
                parse = True
                continue
            if "$$EOE" in line:
                parse = False
                break
            if parse:
                if i % 4 == 0:
                    parts = line.split(" ")
                    time.append([float(parts[0]), parts[3], parts[4]])
                elif i % 4 == 1:
                    parts = line.split("=")[1::]
                    parts = [x.split(" ") for x in parts]
                    position.append([float(parts[0][1]), float(parts[1][0]), float(parts[2][1])])
                elif i % 4 == 2:
                    parts = line.split("=")[1::]
                    parts = [x.split(" ") for x in parts]
                    velocity.append([float(parts[0][1]), float(parts[1][1]), float(parts[2][1])])
                i += 1
    
    return np.array(time), np.array(position), np.array(velocity)

def get_jpl_horizon_time_index(time_arr):
    index = 0
    for i in range(len(time_arr)):
        if time_arr[i][1] == "2018-Jul-14" and time_arr[i][2] == "00:00:00.0000":
            index = i
            break
    return index

def relative_difference(predicted_val, true_val):
    diff = abs(predicted_val - true_val)/true_val
    return diff

def absolute_difference(predicted_val, true_val):
    return abs(predicted_val - true_val)

def usr_orbit_element_errors(predicted_arr, true_arr, keys):
    assert len(predicted_arr) == len(true_arr) == len(keys)
    PERIOD = 396.8146230750213
    predicted_arr[-1] %= PERIOD
    true_arr[-1] %= PERIOD
    rel_diff_arr = []
    for i in range(len(predicted_arr)):
        rel_diff_arr.append(relative_difference(predicted_arr[i], true_arr[i]))
        print(f"{i+1}: Element {keys[i]} || Rel Diff: {round(relative_difference(predicted_arr[i], true_arr[i]) * 100, 3)}% || Abs Diff: {round(absolute_difference(predicted_arr[i], true_arr[i]), 3)}")
    return sum(rel_diff_arr)

def newton_raphson_method(function, derivative, x: float, error_tolerance: float) -> float:
    """
    Newtom Raphson method for approximating root of function with recursion.
    
    :param function: function f(x) to approximate root of as a lambda.
    :param derivative: derivative of f(x) as a lambda.
    :param x: current x
    :param error_tolerance: error tolerance of the approximation.
    :param last_x: default argument to store last_x to check dx sign changes.
    
    :return: approximation of the root.
    """
    next_x = x - (function(x) / derivative(x))
    if abs(next_x - x) <= error_tolerance:
        return x
    else:
        return newton_raphson_method(function, derivative, next_x, error_tolerance)

def taylor_series_sin(x: float, N: int) -> float:
    """
    Taylor series to approximate value of sin(x) about x = 0.
    
    :param x: x to evaluate sin(x) approximation at.
    :param N: number N terms.
    
    :return: approximation of sin(x) given N terms
    """
    x %= 2 * np.pi
    result = 0
    for i in range(int(N)):
        result += (((-1)**i)/(np.math.factorial(2*i + 1))) * (x**(2*i + 1))
    return result
        
def deg_to_rad(angle):
    angle = angle*(np.pi/180)
    return angle

def rad_to_deg(angle):
    angle = angle*(180/np.pi)
    return angle

def ra_to_deg(hours: int, mins: int, seconds: float, deg_status: bool = True) -> float:
    """
    ps1 p13(b): hms -> deg or rad

    :param deg_status: bool: whether deg or rad. if true, then use deg.
    """
    hms_conversion_rate = 15
    mins = (mins/60)
    seconds = (seconds/3600)
    hours = hours + mins + seconds
    # 1 hr = 15 deg
    degree = (hours * hms_conversion_rate) % 360
    if deg_status == True:
        degree = (np.pi/180)*degree
    return degree

def dot_product(vec1, vec2):
    """
    Dot product.
    
    :param vec1, vec2: list[float].
    :return: scalar dot product
    """
    assert(len(vec1) == len(vec2))
    dot = 0
    for i in range(len(vec1)):
        dot += (vec1[i] * vec2[i])
    return dot

def cross_product(vec1, vec2):
    """
    Cross product.
    
    :param vec1, vec2: list[float]. vectors in R^3
    :return: list[float] vector dot product
    """
    assert(len(vec1) == len(vec2))
    assert(len(vec1) == 3)

    ival = ((vec1[1]*vec2[2]) - (vec1[2]*vec2[1]))
    jval = (-(vec1[0]*vec2[2]) + (vec1[2]*vec2[0]))
    kval = ((vec1[0]*vec2[1]) - (vec1[1]*vec2[0]))

    result = [ival,jval,kval]
    return result

def triple_product(vec1, vec2, vec3):
    """
    Triple product.
    
    :param vec1, vec2, vec3: list[float] in R^3
    :return: float. triple product
    """
    vecb = cross_product(vec2, vec3)
    result = dot_product(vec1, vecb)
    return result


def dec_dms_to_deg(degrees: int, arc_minutes: int, arc_seconds: float) -> float:
    """
    DMS to decimal degrees (PS2-6).
    
    :param degrees: Degrees of declination in DMS.
    :param arc_minutes: Arc minutes of declination in DMS.
    :param arc_seconds: Arc seconds of declination in DMS.
    :return: Declination in decimal degrees.
    """
    if degrees < 0.0:
        result = degrees - (arc_minutes/60) - (arc_seconds/3600)
    elif degrees == -0.0:
        result = -1*(arc_minutes/60) - (arc_seconds/3600)
    else:
        result = degrees + (arc_minutes/60) + (arc_seconds/3600)
    return result

def solve_matrix_eqn(coord_vec, transform_matrix):
    params = np.matmul(inv(transform_matrix), coord_vec)
    return params

def ra_deg_to_hms(ra_decimal: float):
    """
    RA from decimal degrees to HMS form (PS2-6).
    
    :param ra_decimal: RA in decimal degrees.
    :return: RA in HMS.
    :rtype: Tuple[int, int, float]
    """
    ra_decimal = (ra_decimal/15.0) % 24.0
    hours = int(ra_decimal)
    if (hours == 0) and (ra_decimal < 0.0):
        hours *= -1.0
    minutes = abs(int((ra_decimal - int(ra_decimal)) * 60))
    seconds = (abs(((ra_decimal - int(ra_decimal)) * 60)) - minutes) * 60
    # print(f"RA: {hours}h {minutes}m {seconds}s ")
    return (hours, minutes, seconds)
    
def dec_deg_to_dms(deg_decimal):
    """
    Declination from decimal degrees to DMS form (PS2-6).
    
    :param deg_decimal: Declination in decimal degrees.
    :return: Declination in DMS.
    :rtype: Tuple[int, int, float]
    """
    degrees = int(deg_decimal)
    if (degrees == 0) and (deg_decimal < 0.0):
        degrees *= -1.0
    arcminutes = abs(int((deg_decimal - int(deg_decimal)) * 60))
    arcseconds = (abs(((deg_decimal - int(deg_decimal)) * 60)) - arcminutes) * 60
    # print(f"DMS: {degrees}° {arcminutes}' {arcseconds}\" ")
    return (degrees, arcminutes, arcseconds)

def least_squares_regression(arr_x, arr_y):
    assert(len(arr_x) == len(arr_y))
    x_sum = sum(arr_x)
    y_sum = sum(arr_y)
    N = len(arr_x)
    xy_sum = 0
    for i in range(N):
        xy_sum += arr_x[i] * arr_y[i]
    x2_sum = 0
    for i in range(N):
        x2_sum += arr_x[i]**2
    a = ((N * xy_sum) - (x_sum * y_sum))/((N * x2_sum) - (x_sum**2))
    b = ((x2_sum * y_sum) - (xy_sum * x_sum))/((N * x2_sum) - (x_sum**2))
    return a,b

def az_alt_to_ra_dec(azimuth: float, altitude: float, lst: tuple[int, int, float]) -> tuple[float, float]:
    """
    Convert azimuth and altitude to RA/Dec given LST and Latitude info (PS3-6).
    
    :param azimuth: Azimuth in degrees.
    :param altitude: Altitude in degrees.
    :param lst: LST in HMS.
    
    :return: tuple of RA, DEC in degrees
    """
    # convert to radians to work with
    azimuth = azimuth * np.pi/180
    altitude = altitude * np.pi/180
    
    # convert to decimals from HMS
    lst = ra_to_deg(*lst, True)
    latitude = 40 * np.pi/180
    
    x = np.sin(altitude) * np.sin(latitude) + np.cos(altitude) * np.cos(latitude) * np.cos(azimuth)
    declination = np.arcsin(x)
    numerator = np.sin(altitude) - (np.sin(latitude) * np.sin(declination))
    denominator = np.cos(latitude) * np.cos(declination)
    ratio = numerator/denominator
    
    if abs(ratio) >= 1: # Round to avoid not in domain of np.arccos
        ratio = np.round(ratio)
    
    # hour angle handling other side
    if azimuth < np.pi:
        right_ascension = lst + np.arccos(ratio)
    else:
        right_ascension = lst - np.arccos(ratio)
        
    # convert back to degrees
    return ((right_ascension * 180/np.pi) % 360, declination * 180/np.pi)


def lspr_check(csv_file: str):
    """
    Pixel (x, y) to RA, DEC - LSPR Practice exercise.
    
    :param csv_file: Data csv file containing data.
    
    :return: (b1 [deg], b2 [deg], a11 [deg/pix], a12 [deg/pix], a21 [deg/pix], a22 [deg/pix], ra [deg], dec [deg], ra_uncert [arcsec], dec_uncert [arcsec])
    :rtype: tuple
    """
    
    # open and parse csv file
    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f)
        data_list = list(csv_data)
        np_array = np.array(data_list)

    # parse data into np arrays
    star_values = np_array[:6]
    asteroid_values = np_array[6][:2]
    
    ra_values = []
    dec_values = []
    for i in range(np.size(star_values, 0)):
        ra_values.append(ra_to_deg(float(star_values[i][2]), float(star_values[i][3]), float(star_values[i][4]), False))
        dec_values.append(dec_dms_to_deg(float(star_values[i][5]), float(star_values[i][6]), float(star_values[i][7])))

    ra_values = np.array(ra_values)
    dec_values = np.array(dec_values)
    
    # slice columns and convert to float from string
    arr_x = star_values[:, 0].astype(np.float64)
    arr_y = star_values[:, 1].astype(np.float64)

    # compute sums
    x_sum = sum(arr_x)
    y_sum = sum(arr_y)
    N = len(arr_x)
    xy_sum = 0
    x2_sum = 0
    y2_sum = 0
    for i in range(N):
        xy_sum += arr_x[i] * arr_y[i]
        x2_sum += arr_x[i]**2
        y2_sum += arr_y[i]**2
    
    # compute transform mapping matrix
    transform_matrix = np.array([[N, x_sum, y_sum], [x_sum, x2_sum, xy_sum], [y_sum, xy_sum, y2_sum]])
    
    # compute sums for vectors
    alpha_sum = sum(ra_values)
    delta_sum = sum(dec_values)
    alphax_sum = 0
    alphay_sum = 0
    deltax_sum = 0
    deltay_sum = 0
    for i in range(N):
        alphax_sum += arr_x[i] * ra_values[i]
        alphay_sum += arr_y[i] * ra_values[i]
        deltax_sum += arr_x[i] * dec_values[i]
        deltay_sum += arr_y[i] * dec_values[i]

    # create vectors for alpha and delta
    RA_vec = np.array([[alpha_sum], [alphax_sum], [alphay_sum]])
    dec_vec = np.array([[delta_sum], [deltax_sum], [deltay_sum]])
    
    # solve for params b and a vector using inverse matrix
    param1 = solve_matrix_eqn(RA_vec, transform_matrix)
    param2 = solve_matrix_eqn(dec_vec, transform_matrix)
    
    # find asteroid RA and DEC
    asteroid_values = asteroid_values.astype(np.float64)
    alpha_asteroid = param1[0][0] + param1[1][0] * asteroid_values[0] + param1[2][0] * asteroid_values[1]
    alpha_asteroid = alpha_asteroid/15
    delta_asteroid = param2[0][0] + param2[1][0] * asteroid_values[0] + param2[2][0] * asteroid_values[1]

    # compute residuals
    ra_residuals = []
    dec_residuals = []
    for i in range(len(arr_x)):
        ra_residuals.append((ra_values[i] - (param1[0][0] + param1[1][0] * arr_x[i] + param1[2][0] * arr_y[i]))*3600)
        dec_residuals.append((dec_values[i] - (param2[0][0] + param2[1][0] * arr_x[i] + param2[2][0] * arr_y[i]))*3600)

    # compute standard deviation
    sigma_ra = 0
    sigma_dec = 0
    for i in range(len(ra_residuals)):
        sigma_ra += (ra_residuals[i])**2
        sigma_dec += (dec_residuals[i])**2
    
    sigma_ra *= 1/(len(ra_residuals) - 3)
    sigma_dec *= 1/(len(dec_residuals) - 3)
    sigma_ra = np.sqrt(sigma_ra)
    sigma_dec = np.sqrt(sigma_dec)
    
    return (param1[0][0], param2[0][0], param1[1][0], param1[2][0], param2[1][0], param2[2][0], alpha_asteroid, delta_asteroid, sigma_ra, sigma_dec)


def integrate(integrand, lower_bound: int, upper_bound: int, N: int) -> float:
    """
    Numerical integrator (PS2-4). 
    (Note: Min N for an accurate answer with midpoint Riemann sum is N = 4*1e4.)
    
    :param integrand: Function to integrate f(x) as lambda.
    :param lower_bound: Lower bound of integral.
    :param upper_bound: Upper bound of integral.
    :param N: Number of intervals.
    :return: Estimate of integral with midpoint Riemann sum.
    """
    interval = (upper_bound - lower_bound)/N
    x = lower_bound
    result = 0
    while (x + interval) <= upper_bound: # is "<=" not "<" (matters for N=1 case)
        if x == lower_bound:
            x += interval/2
            result += interval * integrand(x)
        else:
            x += interval
            result += interval * integrand(x)
    return result

def sin_cos_to_angle(sine_value: float, cosine_value: float) -> float:
    """
    Determine angle in unit circle in radians from sine and cosine value (PS2-5).
    
    :param sine_value: Value of sin(desired_angle).
    :param cosine_value: Value of cos(desired_angle).
    :return: Desired angle in radians.
    """
    angle = np.arccos(cosine_value) # returns angle in [0, pi]
    if sine_value < 0:
        angle += 2*(np.pi - angle)
    return angle

def magnitude(vec) -> float:
    """
    Find magnitude of a vector (PS2-6).
    
    :param vec: np.ndarray[float64]. Input vector.
    :return: Magnitude of vector.
    """
    result = 0
    for i in range(vec.size):
        result += (vec[i])**2 
    result = np.sqrt(result)
    return result

def rotate_vector(vec, alpha: float, beta: float):
    """
    Rotate vector about z-axis by alpha CW and then x-axis by beta CW (PS2-7).
    
    :param vec: Vector to rotate (row vectors).
    :param alpha: Rotate angle clockwise about z-axis.
    :param beta: Rotate angle clockwise about x-axis.
    :return: Rotated vector.
    :rtype: np.ndarray[np.float64]
    """
    rotation_matrix = np.array([[np.cos(alpha), np.sin(alpha), 0.0],
                                [-1*np.cos(beta)*np.sin(alpha), np.cos(beta)*np.cos(alpha), np.sin(beta)],
                                [np.sin(alpha)*np.sin(beta), -1*np.sin(beta)*np.cos(alpha), np.cos(beta)]])
    result = np.matmul(rotation_matrix, np.transpose(vec))
    return result


