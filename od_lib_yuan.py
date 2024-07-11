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
        return np.sqrt(1 - ((specific_momentum_mag**2)/(semi_major_axis)))

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
		self.earth_sun_vector = earth_sun_vector.astype(np.float64)
		self.time = np.array([time]).astype(np.float64)[0]

		assert earth_sun_vector.shape[0] == 3

		# init constants
		self.K_CONSTANT = 0.0172020989484
		self.MU = 1.0
		self.EARTH_TILT = np.radians(23.4392911)

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

# class GaussMethod




