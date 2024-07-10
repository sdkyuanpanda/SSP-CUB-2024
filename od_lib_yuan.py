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
    def __init__(self, 
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

        
# class GaussMethod




