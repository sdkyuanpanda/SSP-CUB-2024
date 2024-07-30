"""
Main Orbital Determination (OD) Library Code.
"""

import numpy as np
import helper_od_yuan as helper

class OrbitalDetermination:
    """
    Utilities to get orbit parameters (a, e, i, Omega, omega, anomalies) 
    from position and velocity vectors (r and rdot) and time (for certain anomaly calculation only).
    
    Orbital parameters to get:
        a: semi-major axis
        e: eccentricity
        i: inclination
        omega: longitude of the ascending node
        Omega: argument of perihelion
        nu: true anomaly (or mean or eccentric anomaly)

    :param position_vec: position vector r [AU].
    :param velocity_vec: velocity vector rdot [AU/Gaussian Days].
    :param time: time little t at measured postiion_vec [Julian Days]
    """
    def __init__(
	self, 
        position_vec: np.ndarray[np.float64], 
        velocity_vec: np.ndarray[np.float64],
        time: float
    ):
        assert position_vec.shape[0] == velocity_vec.shape[0] == 3
        
        # ensure float64 arrays for accuracy
        self.position_vec = position_vec.astype(np.float64)
        self.velocity_vec = velocity_vec.astype(np.float64)
        self.time = time.astype(np.float64)

        # calculate magnitudes
        self.position_mag = np.linalg.norm(self.position_vec)
        self.velocity_mag = np.linalg.norm(self.velocity_vec)
        
        # define constants
        self.GAUSSIAN_YEAR = 365.2568983
        self.K_CONSTANT = 0.0172020989484
        self.LIGHT_SPEED = 173.144643267
        self.EARTH_TILT = np.radians(23.4374)

    def _calc_specific_momentum(self) -> np.ndarray[np.float64]:
        """
        Specific angular momentum vector h from r and rdot vectors with cross product.

            h = r x rdot

        :return: specific angular momentum vector.
        """
        return np.cross(self.position_vec, self.velocity_vec)
    
    def calc_semi_major_axis(self) -> float:
        """
        Calculate semi-major axis a from r and rdot magnitudes using vis-viva.

            a = 1/((2/r) - (v^2))
        
        :return: semi-major axis value.
        """
        return 1/((2/self.position_mag) - (self.velocity_mag**2))

    def calc_eccentricity(self) -> float:
        """
        Calculate eccentricity with:

            e = sqrt{1 - (h^2/a)}

        :return: eccentricity value.
        """
        specific_momentum_mag = np.linalg.norm(self._calc_specific_momentum())
        semi_major_axis = self.calc_semi_major_axis()
        eccentricity = np.sqrt(1 - ((specific_momentum_mag**2)/(semi_major_axis)))
        assert 0 <= eccentricity < 1
        return eccentricity

    def calc_inclination(self) -> float:
        """
        Calculate inclination with specific momentum:

            cos i = h_z / h

        :return: inclination value [rad].
        """
        specific_momentum_vec = self._calc_specific_momentum()
        specific_momentum_mag = np.linalg.norm(specific_momentum_vec)
        return np.arccos(specific_momentum_vec[2]/specific_momentum_mag)

    def calc_asc_node_longitude(self) -> float:
        """
        Calculate longitude of ascending node with:

            sin Omega = h_x/(h sin i)
            cos Omega = -h_y/(h sin i)

        :return: longitude of ascending node [rad].
        """
        specific_momentum_vec = self._calc_specific_momentum()
        specific_momentum_mag = np.linalg.norm(specific_momentum_vec)
        
        denom_holder = specific_momentum_mag * np.sin(self.calc_inclination())
        
        # calculate sine and cosine values
        sine_value = specific_momentum_vec[0]/(denom_holder) 
        cosine_value = -1 * specific_momentum_vec[1]/(denom_holder)
        
        return helper.sin_cos_to_angle(sine_value, cosine_value)

    def _calc_true_anomaly(self) -> float:
        """
        Calculate true anomaly with:

            sin nu = (a(1-e^2) r dot rdot)/(he r)
            cos nu = 1/e[(a(1-e^2))/r - 1]

        :return: true anomaly [rad].
        """
        # get other params
        semi_major_axis = self.calc_semi_major_axis()
        eccentricity = self.calc_eccentricity()
        specific_momentum_mag = np.linalg.norm(self._calc_specific_momentum())
    
        # get true anomaly nu
        sine_nu = ((semi_major_axis * (1 - (eccentricity**2))) / (specific_momentum_mag * eccentricity)) * (np.dot(self.position_vec, self.velocity_vec) / self.position_mag)
        cosine_nu = (1/eccentricity) * (((semi_major_axis * (1-(eccentricity**2)))/(self.position_mag)) - 1)
        true_anomaly = helper.sin_cos_to_angle(sine_nu, cosine_nu)
        return true_anomaly

    def calc_perihelion_argument(self) -> float:
        """
        Calculate argument of perihelion with:

            sin U = z/(r sin i)
            cos U = (x cos Omega + y sin Omega)/r
            omega = U - nu
        
        :return: argument of perihelion [rad].
        """
        # get other params
        inclination = self.calc_inclination()
        asc_node_longitude = self.calc_asc_node_longitude()

        # get n hat and r hat unit vectors
        n_uvec = np.array([np.cos(asc_node_longitude), np.sin(asc_node_longitude), 0])
        r_uvec = self.position_vec/self.position_mag

        # compute z
        z_comp = (np.cross(n_uvec, r_uvec)[0]) * self.position_mag * (1/np.sin(asc_node_longitude))

        # get U
        sine_U = z_comp/(self.position_mag * np.sin(inclination))
        cosine_U = (np.dot(self.position_vec, n_uvec))/self.position_mag
        U_angle = helper.sin_cos_to_angle(sine_U, cosine_U)

        # get true anomaly nu
        true_anomaly = self._calc_true_anomaly()

        # get perihelion argument omega
        perihelion_argument = (U_angle - true_anomaly) % (2*np.pi)

        return perihelion_argument

    def calc_anomalies(self) -> tuple[float, float]:
        """
        Calculate mean anomaly M and time of perihelion passage T with:
            
            M = E - esinE
            T = t - M/n

        :return: mean anomaly M [rad], time of perihelion passage T [Julian days]
        """
        # get other params
        semi_major_axis = self.calc_semi_major_axis()
        eccentricity = self.calc_eccentricity()
        true_anomaly = self._calc_true_anomaly()

        # calculate eccentric anomaly E
        sine_E = (self.position_mag * np.sin(true_anomaly)) / (semi_major_axis * np.sqrt(1 - (eccentricity**2)))
        cosine_E = ((semi_major_axis * eccentricity) + (self.position_mag * np.cos(true_anomaly)))/semi_major_axis

        eccentric_anomaly = helper.sin_cos_to_angle(sine_E, cosine_E)

        # calculate mean anomaly M
        mean_anomaly = eccentric_anomaly - (eccentricity * np.sin(eccentric_anomaly))

        # calculate time of perihelion passage T
        period = (2*np.pi) * (semi_major_axis**(1.5)) * (1/(self.K_CONSTANT))
        perihelion_passage_time = self.time - (mean_anomaly * semi_major_axis**(1.5) * 1/(self.K_CONSTANT))
        
        return (mean_anomaly, perihelion_passage_time)
    
    def get_orbital_elements(self, use_radians: bool = True):
        """
        Main function to get all orbital elements.

        :return: tuple with all orbital elements (including M and T); all floats.
                 (a, e, i, Omega, omega, M, T)
        """
        M, T = self.calc_anomalies()
        if use_radians:
            return np.array([self.calc_semi_major_axis(), self.calc_eccentricity(), self.calc_inclination(), self.calc_asc_node_longitude(), self.calc_perihelion_argument(), M, T])
        else:
            return np.array([self.calc_semi_major_axis(), self.calc_eccentricity(), helper.rad_to_deg(self.calc_inclination()), helper.rad_to_deg(self.calc_asc_node_longitude()), helper.rad_to_deg(self.calc_perihelion_argument()), helper.rad_to_deg(M), T])

class EphemerisGeneration:
	"""
	Ephemeris generation from classical orbital elements and time.

	:param semi_major_axis: semi major axis of orbit a [au].
	:param eccentricity: eccentricity of orbit e.
	:param inclination: inclination of orbit [rad].
	:param asc_node_longitude: longitude of ascending node [rad].
	:param perihelion_argument: perihelion argument [rad].
	:param perihelion_passage_time: perihelion passage time [julian days]
	:param earth_sun_vector: earth sun vector - np.ndarray[np.float64] [au]
	:param time: time of ephemeris to generate [julian days].
	"""
	def __init__(
		self,
		semi_major_axis: float,
		eccentricity: float,
		inclination: float,
		asc_node_longitude: float,
		perihelion_argument: float,
		perihelion_passage_time: float,
		earth_sun_vector: np.ndarray[np.float64],
		time: float
	):
		# init orbital elements
		self.semi_major_axis = semi_major_axis.astype(np.float64)
		self.eccentricity = eccentricity.astype(np.float64)
		self.inclination = inclination.astype(np.float64)
		self.asc_node_longitude = asc_node_longitude.astype(np.float64)
		self.perihelion_argument = perihelion_argument.astype(np.float64)
		self.perihelion_passage_time = perihelion_passage_time.astype(np.float64)
		self.time = np.array([time]).astype(np.float64)[0]
		
		if earth_sun_vector.any() == None:
			earth_sun_vector = helper.get_sun_earth_vector(time)
		self.earth_sun_vector = earth_sun_vector.astype(np.float64)

		assert self.earth_sun_vector.shape[0] == 3

		# init constants
		self.K_CONSTANT = 0.0172020989484
		self.MU = 1.0
		self.EARTH_TILT = np.radians(23.4374)
		self.LIGHT_SPEED = 173.144643267
		# self.EARTH_TILT = np.radians(23.4392911)

	def _calc_anomalies(self):
		"""
		Compute mean anomaly and use newtons method to compute eccentric anomaly.

		:return: eccentric anomaly, mean anomaly
		:rtype: float, float
		"""
		# compute mean anomaly
		n_constant = self.K_CONSTANT * np.sqrt(self.MU/(self.semi_major_axis**3))
		mean_anomaly = n_constant * (self.time - self.perihelion_passage_time)
		mean_anomaly %= 2*np.pi

		# approximate eccentric anomaly using newton-raphson method
		mean_anomaly_function = lambda x: mean_anomaly - (x - (self.eccentricity * np.sin(x)))
		mean_anomaly_derivative = lambda x: (self.eccentricity * np.cos(x)) - 1
		eccentric_anomaly = helper.newton_raphson_method(
			mean_anomaly_function,
			mean_anomaly_derivative,
			mean_anomaly,
			1e-10
		)
		return eccentric_anomaly, mean_anomaly

	def get_position_vector(self) -> np.ndarray[np.float64]:
		"""
		Compute position vector with eccentric anomaly.

		:return: position vector in orbital plane.
		"""
		eccentric_anomaly, _ = self._calc_anomalies()
		position_vector = np.array([
			((self.semi_major_axis * np.cos(eccentric_anomaly)) - (self.semi_major_axis * self.eccentricity)),
			self.semi_major_axis * np.sqrt(1 - (self.eccentricity**2)) * np.sin(eccentric_anomaly),
			0
		])
		return position_vector
	
	def _calc_orbital_to_ecliptic(self) -> np.ndarray[np.float64]:
		"""
		Private function to calculate rotated position vector in ecliptic plane from orbital.

		:return: position vector in ecliptic plane.
		"""
		position_vector = self.get_position_vector()

		rotation_1 = np.array([
			[np.cos(self.perihelion_argument), -1*np.sin(self.perihelion_argument), 0],
			[np.sin(self.perihelion_argument), np.cos(self.perihelion_argument), 0],
			[0, 0, 1]
		])
		rotation_2 = np.array([
			[1, 0, 0],
			[0, np.cos(self.inclination), -1*np.sin(self.inclination)],
			[0, np.sin(self.inclination), np.cos(self.inclination)]
		])
		rotation_3 = np.array([
			[np.cos(self.asc_node_longitude), -1*np.sin(self.asc_node_longitude), 0],
			[np.sin(self.asc_node_longitude), np.cos(self.asc_node_longitude), 0],
			[0, 0, 1]
		])
		
		rotated_vec_1 = np.matmul(rotation_1, position_vector)
		rotated_vec_2 = np.matmul(rotation_2, rotated_vec_1)
		rotated_vec_3 = np.matmul(rotation_3, rotated_vec_2)

		return rotated_vec_3
	
	def _calc_ecliptic_to_equatorial(self) -> np.ndarray[np.float64]:
		"""
		Private function to calculate rotated position vector from ecliptic to equatorial plane.

		:return: position vector in equatorial plane.
		"""
		position_vector = self._calc_orbital_to_ecliptic()
		rotation = np.array([
			[1, 0, 0],
			[0, np.cos(self.EARTH_TILT), -1*np.sin(self.EARTH_TILT)],
			[0, np.sin(self.EARTH_TILT), np.cos(self.EARTH_TILT)]
		])

		position_vector = np.matmul(rotation, position_vector)
		return position_vector
	
	def _calc_ra_dec(self):
		"""
		Private function to compuate RA and DEC from previously computed values.

		:return: right ascension [rad], declination [rad]
		:rtype: float, float
		"""
		position_vector = self._calc_ecliptic_to_equatorial()
		range_vector = position_vector + self.earth_sun_vector
		unit_range_vector = range_vector/(np.linalg.norm(range_vector))
		
		assert unit_range_vector.shape[0] == 3
		
		# calculate declination
		declination = np.arcsin(unit_range_vector[2])
		
		# calculate right ascension (need quadrant check)
		sine_right_ascension = (unit_range_vector[1])/np.cos(declination)
		cosine_right_ascension = (unit_range_vector[0])/np.cos(declination)
		right_ascension = helper.sin_cos_to_angle(sine_right_ascension, cosine_right_ascension)
		return right_ascension, declination
	
	def get_ra_dec(self, use_radians: bool = True):
		"""
		Public retrieval of right ascension and declination.

		:return: right ascension and declination [rad] - both floats.
		"""
		ra, dec = self._calc_ra_dec()
		if use_radians:
			return np.array([ra, dec])
		else:
			return np.array([np.degrees(ra), np.degrees(dec)])

class GaussMethod:
	"""
	Utilities for Method of Gauss to determine position and velocity vector at middle observation from RA and DEC of 3 observations.

	:param right_ascension: array of 3 right ascension values.
							corresponds to 3 observations.
	:param declination: array of 3 declination values.
						corresponds to 3 observations.
	:param times: array of 3 times that are the times of the 3 observations.
	:param sun_vectors: 2d array of 3 sun vectors at each of the three times.
	"""
	def __init__(
		self,
		right_ascension: np.ndarray[np.float64],
		declination: np.ndarray[np.float64],
		times: np.ndarray[np.float64],
		sun_vectors: np.ndarray[np.float64]
	):
		assert right_ascension.shape[0] == declination.shape[0] == times.shape[0] == 3

		# init params
		self.right_ascension = right_ascension.astype(np.float64)
		self.declination = declination.astype(np.float64)
		self.times = times.astype(np.float64)

		# init consts
		self.K_CONSTANT = 0.0172020989484
		self.MU = 1.0
		self.EARTH_TILT = np.radians(23.4374)
		self.LIGHT_SPEED = 173.144643267 # in AU/solar days

		# init sun vectors
		if sun_vectors.any() == None:
			sun_vectors = np.empty((0,3))
			for i in range(times.shape[0]):
				sun_vectors = np.vstack([sun_vectors, helper.get_sun_earth_vector(times[i])])
		self.sun_vectors = sun_vectors

	def _calc_unit_range_vectors(self) -> np.ndarray[np.float64]:
		"""
		Compute unit range vectors rho hat.

		:return: np.ndarray of 3 unit range vectors rho hat.
		"""
		unit_range_vectors = np.empty((0, 3))
		for i in range(self.right_ascension.shape[0]):
			rho_hat_vec = np.array([
				np.cos(self.right_ascension[i]) * np.cos(self.declination[i]),
				np.sin(self.right_ascension[i]) * np.cos(self.declination[i]),
				np.sin(self.declination[i])
			])
			
			assert abs(np.linalg.norm(rho_hat_vec) - 1.0) < 1e-10

			unit_range_vectors = np.vstack([unit_range_vectors, rho_hat_vec])
		return unit_range_vectors

	def _calc_gaussian_time_intervals(self, new_times) -> np.ndarray[np.float64]:
		"""
		Compute gaussian time intervals tau from adjusted times (days).
		Created new placeholder new_times array to be adjusted (self.times is not incremented in light correction).

		:param new_times: seperate time values from self.times adjusted by light travel time.

		:return: array of tau values [tau1, tau0, tau3] [Gaussian Days]
		"""
		tau1 = self.K_CONSTANT * (new_times[0] - new_times[1])
		tau0 = self.K_CONSTANT * (new_times[2] - new_times[0])
		tau3 = self.K_CONSTANT * (new_times[2] - new_times[1])
		return np.array([tau1, tau0, tau3])
	
	def _calc_initial_a_values(self, tau_values) -> np.ndarray[np.float64]:
		"""
		Compute the initial a values (a1, a3) from tau.

		:param tau_values: np.ndarray of 3 tau values [tau1, tau0, tau3]

		:return: array of intiial a values [a1, a3]
		"""
		a1 = (tau_values[2])/(tau_values[1])
		a3 = -1*(tau_values[0])/(tau_values[1])
		return np.array([a1, a3])

	def _calc_d_values(self, unit_range_vectors):
		"""
		Pre-compute the cross product "d" values for easier computation of rho values.

		:param unit_range_vectors: np.ndarray of unit range vectors

		:return: pre-computed d values.
		:rtype: float, np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]
		"""
		# compute d values
		d0_value = np.dot(
			unit_range_vectors[0],
			np.cross(
				unit_range_vectors[1],
				unit_range_vectors[2]
			)
		)
		d1_values = []
		d2_values = []
		d3_values = []
		for j in range(3):
			d1_values.append(
				np.dot(
					np.cross(self.sun_vectors[j], unit_range_vectors[1]), 
					unit_range_vectors[2]
				)	
			)
			d2_values.append(
				np.dot(
					np.cross(unit_range_vectors[0], self.sun_vectors[j]), 
					unit_range_vectors[2]
				)	
			)
			d3_values.append(
				np.dot(
					np.cross(unit_range_vectors[1], self.sun_vectors[j]), 
					unit_range_vectors[0]
				)	
			)
		return d0_value, np.array(d1_values), np.array(d2_values), np.array(d3_values)

	def get_d_values(self) -> np.ndarray[np.float64]:
		"""
		Public retrieval of d values for testing.

		:return: np.ndarray of d values [d0, *d1, *d2, *d3]
		"""
		d0_value, d1_values, d2_values, d3_values = self._calc_d_values()
		return np.array([d0_value, *d1_values, *d2_values, *d3_values])

	def _calc_rho_values(self, a_values, unit_range_vectors) -> np.ndarray[np.float64]:
		"""
		Compute rho values using precomputed d values and other params.

		:param a_values: np.ndarray of a values [a1, a3]
		:param unit_range_vectors: np.ndarray of 3 unit range vectors.

		:return: np.ndarray of rho values [rho1, rho2, rho3]
		"""
		# init D values
		d0_value, d1_values, d2_values, d3_values = self._calc_d_values(unit_range_vectors)

		# calc rho1
		rho_1_num_1 = a_values[0] * d1_values[0]
		rho_1_num_2 = -1 * d1_values[1]
		rho_1_num_3 = a_values[1] * d1_values[2]
		rho_1_dem = a_values[0] * d0_value
		rho_1 = (rho_1_num_1 + rho_1_num_2 + rho_1_num_3)/(rho_1_dem)

		# calc rho2
		rho_2_num_1 = a_values[0] * d2_values[0]
		rho_2_num_2 = -1 * d2_values[1]
		rho_2_num_3 = a_values[1] * d2_values[2]
		rho_2_dem = -1 * d0_value
		rho_2 = (rho_2_num_1 + rho_2_num_2 + rho_2_num_3)/(rho_2_dem)

		# calc rho3
		rho_3_num_1 = a_values[0] * d3_values[0]
		rho_3_num_2 = -1 * d3_values[1]
		rho_3_num_3 = a_values[1] * d3_values[2]
		rho_3_dem = a_values[1] * d0_value
		rho_3 = (rho_3_num_1 + rho_3_num_2 + rho_3_num_3)/(rho_3_dem)

		return np.array([rho_1, rho_2, rho_3])

	def _calc_general_r_vectors(self, unit_range_vectors, a_values, rho_values) -> np.ndarray[np.float64]:
		"""
		Compute 3 r_vectors using rho, unit range, and sun vectors.

		:param unit_range_vectors: np.ndarray of the unit range vectors
		:param a_values: np.ndarray of a values [a1, a3]
		:param rho_values: np.ndarray of 3 rho values [rho1, rho2, rho3]
		
		:return: np.ndarray of position (r_vectors)
		"""
		# calculate r vectors
		r_vectors = (rho_values[:, np.newaxis] * unit_range_vectors) - self.sun_vectors
		return r_vectors
	
	def _calc_initial_r2_dot(self, r_vectors) -> np.ndarray[np.float64]:
		"""
		Compute initial r2 dot vector (middle velocity vector) approximation.

		:param r_vectors: np.ndarray of r_vectors [[r1], [r2], [r3]]

		:return: np.ndarray of initial r2 dot velocity vector.
		"""
		# compute sub vels
		vel12 = (r_vectors[1] - r_vectors[0])/(self.K_CONSTANT * (self.times[1] - self.times[0]))
		vel23 = (r_vectors[2] - r_vectors[1])/(self.K_CONSTANT * (self.times[2] - self.times[1]))

		# compute initial r2dot
		initial_r2dot = (((self.K_CONSTANT * (self.times[2] - self.times[1]))*vel12)+((self.K_CONSTANT * (self.times[1] - self.times[0]))*vel23))/(self.K_CONSTANT * (self.times[2] - self.times[0]))
		return initial_r2dot

	def _calc_fg_values(self, tau, r2, r2_dot) -> np.ndarray[np.float64]:
		"""
		Calculate f and g values with 4th order functions with r2 and r2dot vectors.
		
		:param tau: tau value to calculate fg at
		:param r_vectors: np.ndarray of 3 r vectors
		:param r2_dot: np.ndarray of r2dot vector

		:return: np.ndarray of f and g values [f_value, g_value]
		"""
		# precompute r values
		r2_mag = np.linalg.norm(r2)
		r2_prod_one = np.dot(r2, r2_dot)
		r2_prod_two = np.dot(r2_dot, r2_dot)

		# compute f
		f_value_term_1 = 1 - ((tau**2)/(2*(r2_mag**3)))
		f_value_term_2 = (((tau**3)*(r2_prod_one))/(2*(r2_mag**5)))
		f_value_term_3 = ((tau**4)/(24*(r2_mag**3))) * (3*(((r2_prod_two)/(r2_mag**2))-(1/(r2_mag**3)))-(15*(((r2_prod_one)/(r2_mag**2))**2))+(1/(r2_mag**3)))
		f_value = f_value_term_1 + f_value_term_2 + f_value_term_3
		
		# compute g
		g_value_term_1 = tau - ((tau**3)/(6*(r2_mag**3)))
		g_value_term_2 = (r2_prod_one * (tau**4))/(4*(r2_mag**5))
		g_value = g_value_term_1 + g_value_term_2
		
		return np.array([f_value, g_value])

	def _calc_general_a_values(self, fg_values_1, fg_values_3) -> np.ndarray[np.float64]:
		"""
		Calculate general a values with f and g.

		:param fg_values_1: np.ndarray values of f and g for 1 [f1, g1]
		:param fg_values_3: np.ndarray values of f and g for 3 [f3, g3]

		:return: np.ndarray of a values [a1, a3]
		"""
		a_dem = (fg_values_1[0] * fg_values_3[1]) - (fg_values_3[0] * fg_values_1[1])
		a1 = (fg_values_3[1])/(a_dem)
		a3 = -1*(fg_values_1[1])/(a_dem)
		return np.array([a1, a3])
	
	def get_position_and_velocity(self, use_ecliptic: bool = True):
		"""
		Public retrieval of position and velocity vectors (r2, r2_dot).

		Main loop of Method of Gauss.

		:param use_ecliptic: user-decided use of ecliptic coordinates for position and velocity [default = True]
							 otherwise is in equatorial coordinates (which is used for rest of calculation in Method of Gauss).

		:return: position and velocity vectors (r2, r2_dot) of asteroid at middle observation (default in ecliptic coords).
		:rtype: np.ndarray[np.float64], np.ndarray[np.float64]
		"""
		# init params
		unit_range_vectors = self._calc_unit_range_vectors()
	
		new_times = self.times
		tau_values = self._calc_gaussian_time_intervals(new_times) # tau is: [tau1, tau0, tau3]

		# calc initial values
		a_values = self._calc_initial_a_values(tau_values)
		rho_values = self._calc_rho_values(a_values, unit_range_vectors)

		r_vectors = self._calc_general_r_vectors(unit_range_vectors, a_values, rho_values)
		r2 = r_vectors[1]
		
		r2_dot = self._calc_initial_r2_dot(r_vectors)
		
		fg_values_1 = self._calc_fg_values(tau_values[0], r2, r2_dot)
		fg_values_3 = self._calc_fg_values(tau_values[2], r2, r2_dot)
		
		a_values = self._calc_general_a_values(fg_values_1, fg_values_3)

		# main loop
		last_r2_mag = 0
		counter = 0
		while abs((np.linalg.norm(r2) - last_r2_mag)/(np.linalg.norm(r2))) >= 1e-10:
			last_r2_mag = np.linalg.norm(r2)
		
			# to check non-converging case of very sensitive input values
			counter += 1
			if counter > 5e2:
				break

			# compute new rho values
			rho_values = self._calc_rho_values(a_values, unit_range_vectors)

			# speed of light travel time correction
			new_times = self.times - ((rho_values)/(self.LIGHT_SPEED))
			tau_values = self._calc_gaussian_time_intervals(new_times)

			# compute new r vectors
			r_vectors = self._calc_general_r_vectors(unit_range_vectors, a_values, rho_values)

			# refine new r2 and r2 dot
			r2 = ((fg_values_3[1] * r_vectors[0])-(fg_values_1[1] * r_vectors[2]))/((fg_values_1[0] * fg_values_3[1])-(fg_values_3[0] * fg_values_1[1]))
			r2_dot = ((fg_values_3[0] * r_vectors[0])-(fg_values_1[0] * r_vectors[2]))/((fg_values_3[0] * fg_values_1[1])-(fg_values_1[0] * fg_values_3[1]))
			
			# compute new fg
			fg_values_1 = self._calc_fg_values(tau_values[0], r2, r2_dot)
			fg_values_3 = self._calc_fg_values(tau_values[2], r2, r2_dot)

			# compute new a
			a_values = self._calc_general_a_values(fg_values_1, fg_values_3)
	
		rotation_to_ecliptic = np.array([
			[1, 0, 0],
			[0, np.cos(self.EARTH_TILT), np.sin(self.EARTH_TILT)],
			[0, -1*np.sin(self.EARTH_TILT), np.cos(self.EARTH_TILT)]
		])

		if use_ecliptic:
			r2 = np.matmul(rotation_to_ecliptic, r2)
			r2_dot = np.matmul(rotation_to_ecliptic, r2_dot)
			return r2, r2_dot
		else:
			return r2, r2_dot



		



