import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2

# define units
meters = 1
centimeters = 1e-2 * meters
nanometers = 1e-9 * meters
degrees = np.pi / 180

class IncidentWave():
    def __init__(self, lambda_0=2, theta=0, phi=0, a_TE=1, a_TM=0):
        '''
        Set up the source of the incident light. The schematic of the set up can be found in
        the help page. The direction of light source is default from -z to z.

        Arguments:
            lambda_0: wavelength in the free space
            theta: inclination angle, i.e. incident angle. Unit: degree
            phi: azimuth angle, i.e. rotation of the incident direction w.r.t. the z axis.
                phi == 0 means the k-vector is x-z plane. Unit: degree
            a_TE: amplitude of TE component, the TE wave is defined as polarization parallel
                to y-direction.
            a_TM: amplitude of TM component.

        Example:
            1. phi == 0, theta == 0, a_TE = 1, a_TM = 0: normal incident light
                with the linear polarization in y-direction
            2. phi == 0, theta == 45, a_TE == 1, a_TM == 0: TE wave with incident
                angle of 45 degree
        '''
        self.lambda_0 = lambda_0 # free space wavelength
        self.theta = theta * degrees
        self.phi = phi * degrees
        self.a_TE = a_TE
        self.a_TM = a_TM
        self.k_0 = 2 * np.pi / self.lambda_0
        # define or compute TE and TM polarization
        # n_hat as normal vector, then
        # v_TE = cross(n_hat, k_inc) / abs(cross(n_hat, k_inc))
        # v_TM = cross(k_inc, v_TE) / abs(cross(k_inc, v_TE))
        # pol = [px, py, pz] = a_TE * v_TE + a_TM + v_TM (a is for amplitude)
        if theta == 0:
            self.v_TE = np.array([0,1,0]) # y-polarized
            self.v_TM = np.array([1,0,1])
        else:
            self.k_dir = np.array([np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)])
            self.n = np.array([0, 0, 1])
            self.v_TE = np.cross(self.n, self.k_dir) / self._norm_l2(np.cross(self.n, self.k_dir))
            self.v_TM = np.cross(self.k_dir, self.v_TE) / self._norm_l2(np.cross(self.k_dir, self.v_TE))

        # print(self.v_TE, self.v_TM)
        self.pol = self.a_TE * self.v_TE + self.a_TM * self.v_TM
        self.pol = np.array(self.pol)

    def _norm_l2(self, vec):
        '''
        Calculate L2 norm of a vector
        '''
        return np.sqrt(np.sum(vec**2))


class Layer():
    def __init__(self, er=6.0, ur=1.0, er_0=2, ur_0=1, thickness=0.5*centimeters, Lx=1.75*centimeters, Ly=1.50*centimeters, pattern=None):
        '''
        Set up the layer of with certain pattern and materials. The input should be images with value [0, 1].
        The pattern image is suggested to be added in a later step.

        Arguments:
            er: permittivity of the material
            ur: permeability of the material
            er_0: permittivity of the enviroment
            ur_0: permeability of the enviroment
            thickness: thickness of the layer
            Lx: structure period in x dimension
            Ly: structure period in y dimension
            pattern: image with value from [0, 1]. The structure permittivity will be calculated as
                pattern * er
        '''

        # er and ur for structure
        self.er = er
        self.ur = ur

        # er and ur for media
        self.er_0 = er_0
        self.ur_0 = ur_0

        self.d = thickness
        self.Lx = Lx
        self.Ly = Ly
        self.pattern = None
        if pattern:
            self.add_pattern(pattern)

        self.Nx = -1
        self.Ny = -1
        self.er_mat = None
        self.ur_mat = None
        self.er_convmat = None
        self.ur_convmat = None

    def add_pattern(self, imag, Nx=None, Ny=None, invert=True):

        '''
        Add the pattern represented by images into the layer. Image can either be [0,1]
        or [0,255]. The pixel size of the pattern can be adjusted by interpolation.

        Arguments:
            imag: image representing the structure pattern
            Nx: pixel number in x dimension, if not set, will be the dimension of the image
            Ny: pixel number in y dimension, if not set, will be the dimension of the image
            invert: if True, the 1 and 0 in image will be inverted
        '''
        if np.max(imag) > 200:
            imag = imag / 255.0

        if not Nx or not Ny:
            Nx, Ny = imag.shape
        self.Nx, self.Ny = Nx, Ny

        if invert == True:
            imag = 1 - imag

        imag = cv2.resize(imag, (Ny, Nx), interpolation=cv2.INTER_LINEAR)
        self.pattern = imag


    def plot_pattern(self, options='pattern'):
        '''
        Plot the pattern of the structure
        '''
        if options == 'pattern':
            plt.imshow(np.transpose(self.pattern), cmap='gray')
        elif options == 'er_mat':
            plt.imshow(np.transpose(np.abs(self.er_mat)), cmap='gray')
        elif options == 'ur_mat':
            plt.imshow(np.transpose(np.abs(self.ur_mat)), cmap='gray')
        elif options == 'er_convmat':
            plt.imshow(np.transpose(np.abs(self.er_convmat)))
        elif options == 'ur_convmat':
            plt.imshow(np.transpose(np.abs(self.ur_convmat)))
        plt.show()


    def convmat(self, A, P, Q=1, R=1):
        '''
        Compute the convolution matrix
        '''
        if len(A.shape) == 3:
            pass
        elif len(A.shape) == 2:
            A = A.reshape((A.shape[0], A.shape[1], 1))
        elif len(A.shape) == 1:
            A = A.reshape((A.shape[0], 1, 1))
        Nx, Ny, Nz = A.shape

        # indices of spatial harmonics
        num_harms = P * Q * R
        p = np.array(range(-int(np.floor(P/2)), int(np.floor(P/2))+1))
        q = np.array(range(-int(np.floor(Q/2)), int(np.floor(Q/2))+1))
        r = np.array(range(-int(np.floor(R/2)), int(np.floor(R/2))+1))

        # compute fourier coefficient of A
        A_ft = np.fft.fftn(A, axes=(0,1,2))
        A_ft = np.fft.fftshift(A_ft) / (Nx * Ny * Nz)

        # compute array indices of center harmonic
        p0 = 0 + int(np.floor(Nx/2))
        q0 = 0 + int(np.floor(Ny/2))
        r0 = 0 + int(np.floor(Nz/2))

        # initialize convolution matrix
        C = np.zeros((num_harms, num_harms)).astype(np.complex)

        for r_row in range(R):
            for q_row in range(Q):
                for p_row in range(P):
                    row = (r_row - 0) * Q * P + (q_row - 0) * P + p_row
                    for r_col in range(R):
                        for q_col in range(Q):
                            for p_col in range(P):
                                col = (r_col - 0) * Q * P + (q_col - 0) * P + p_col
                                pfft = p[p_row] - p[p_col]
                                qfft = q[q_row] - q[q_col]
                                rfft = r[r_row] - r[r_col]
                                C[row, col] = A_ft[p0+pfft, q0+qfft, r0+rfft]
        return C

    def build(self):
        '''
        Build the layer to construct essential matrices
        '''

        self.er_mat = (self.er - self.er_0) * self.pattern + self.er_0
        self.ur_mat = (self.ur - self.ur_0) * self.pattern + self.ur_0
        self.er_mat.astype(np.complex)
        self.ur_mat.astype(np.complex)

    def build_convmat(self, P, Q):
        '''
        Build convolution matrices
        '''
        self.er_convmat = self.convmat(self.er_mat, P, Q)
        self.ur_convmat = self.convmat(self.ur_mat, P, Q)


class ModelConfig():
    def __init__(self, er_ref=2, ur_ref=1.0, er_trn=9.0, ur_trn=1.0, Lx=1.75*centimeters, Ly=1.50*centimeters, layers=[]):
        '''
        Construct the simulation model with various layers

        Arguments:
            er_ref: permittivity of the incident side
            ur_ref: permeability of the incident side
            er_trn: permittivity of the transmision side
            ur_trn: permittivity of the transmision side
            Lx: model dimension in x direction
            Ly: model dimension in y direction
            layers: Layer() class

        The dimension Lx and Ly should be consistent with each layer. The layer from 0 ... to len(layers)
        represent the layer from bottom to top.

        After simulation, the transmittance and reflectance are stored in the ModelConfig class:
            tx, ty, tz: x, y, and z components of the amplitude transmittance. np.matrix
            rx, ry, rz: x, y, and z components of the amplitude reflectance. np.matrix
            T: intensity transmittances of all orders. np.matrix
            R: intensity reflectance of all orders. np.matrix
            T_tot: sum of all orders of transmitted lights. scalar
            R_tot: sum of all orders of relfected lights. saclar. T_tot + R_tot should be 1.
        '''
        # in reflection region
        self.er_ref = er_ref ### set to 2
        self.ur_ref = ur_ref

        # in transmission region
        self.er_trn = er_trn
        self.ur_trn = ur_trn

        # layers
        self.Lx = Lx
        self.Ly = Ly
        self.layers = []

        self.num_layers = len(self.layers)

        # simulation results are updated here
        self.tx, self.ty, self.tz = None, None, None
        self.rx, self.ry, self.rz = None, None, None

        self.T, self.R = None, None
        self.T_tot, self.R_tot = None, None

    def add_layer(self, layer):
        '''

        '''
        self.layers.append(layer)
        self.num_layers += 1

    def build(self):
        self.num_layers = 0
        for i in range(len(self.layers)):
            self.layers[i].build()
            self.num_layers += 1




class Simulation():
    def __init__(self, model, source, x_harms=3, y_harms=3):
        '''
        Simulation module, with the model and source class

        Arguments:
            model: model to simulation. ModelConfig class
            source: light source. IncidentWave class
            x_harms: expansion of fft in x direction
            y_harsm: expansion of fft in y direction
        '''

        self.model = model
        self.source = source
        self.x_harms = x_harms
        self.y_harms = y_harms

        self.S_device = None
        self.S_global = None

        # self.tx, self.ty, self.tz = None, None, None
        # self.rx, self.ry, self.rz = None, None, None

        # self.T, self.R = None, None
        # self.T_tol, self.R_tol = None, None

    def run_simulation(self):
        '''
        Run the simulation, the resutls are automatically updated in the model class
        '''

        # 1. compute covolution matrices
        for layer in self.model.layers:
            layer.build_convmat(self.x_harms, self.y_harms)

        # 2. compute wave vector expansion
        Kx, Ky, Kz, Kz_ref, Kz_trn = self._wavevec_expansion()

        # 3. compute eigen-modes of free space
        Q_0, V_0, W_0, eigen_vals = self._eigen_freespace(Kx, Ky, Kz)

        # 4. initialize device S-matrix
        self.S_device = self.init_device_S_mat()

        # 5. main loop iteration through layers
        self.S_device = self._device_S_mat(self.S_device, Kx, Ky, W_0, V_0)
        # print(self.S_device[2][0])

        # 6. compute reflection and transmission side S-matrix
        S_ref, W_ref = self._reflection_S_mat(Kx, Ky, Kz_ref, W_0, V_0)
        S_trn, W_trn = self._transmission_S_mat(Kx, Ky, Kz_trn, W_0, V_0)

        # 7. compute global scattering Matrix
        self.S_global = self.RedhefferProd(S_ref, self.S_device)
        self.S_global = self.RedhefferProd(self.S_global, S_trn)

        # 8. compute transmittance and reflectance
        self.update_T_R(Kx, Ky, Kz_ref, Kz_trn, W_ref, W_trn)


    def update_T_R(self, Kx, Ky, Kz_ref, Kz_trn, W_ref, W_trn):
        '''
        Calculate the transmittances and reflectances
        '''
        # compute source paramters
        xy_harms = self.x_harms * self.y_harms
        delta = [0] * xy_harms
        delta[int(np.floor(xy_harms/2))] = 1
        delta = np.array(delta)

        E_pol = self.source.pol
        E_src_x = E_pol[0] * delta
        E_src_y = E_pol[1] * delta
        E_src = np.concatenate((E_src_x, E_src_y), 0)

        # compute source modal coeffients
        C_src = np.matmul(np.linalg.inv(W_ref), E_src)

        # compute transmission and reflection modal coefficients
        C_ref = np.matmul(self.S_global[0], C_src.transpose())
        C_trn = np.matmul(self.S_global[2], C_src.transpose())

        # compute reflected and transmitted fields
        E_ref = np.matmul(W_ref, C_ref)
        E_trn = np.matmul(W_trn, C_trn)

        rx = E_ref[:len(E_ref)//2]
        ry = E_ref[len(E_ref)//2:]
        tx = E_trn[:len(E_trn)//2]
        ty = E_trn[len(E_trn)//2:]
        rz = -np.linalg.inv(Kz_ref) * (np.matmul(Kx, rx) + np.matmul(Ky, ry))
        tz = -np.linalg.inv(Kz_trn) * (np.matmul(Kx, tx) + np.matmul(Ky, ty))

        rx = np.squeeze(np.asarray(rx))
        ry = np.squeeze(np.asarray(ry))
        rz = np.squeeze(np.asarray(rz))

        tx = np.squeeze(np.asarray(tx))
        ty = np.squeeze(np.asarray(ty))
        tz = np.squeeze(np.asarray(tz))

        # step 13: compute diffraction efficiencies
        # compute reflected power
        n_ref = np.complex(np.sqrt(self.model.er_ref))
        n_trn = np.complex(np.sqrt(self.model.er_trn))
        k_0 = self.source.k_0
        theta = self.source.theta
        phi = self.source.phi
        k_inc = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) * n_ref

        r_sqr = np.abs(rx)**2 + np.abs(ry)**2 + np.abs(rz)**2
        R = np.matmul(np.real(0 - Kz_ref / self.model.ur_ref) / np.real(k_inc[2] / self.model.ur_ref),  r_sqr)
        R = R.reshape((self.x_harms, self.y_harms)).transpose() # why transpose???

        t_sqr = np.abs(tx)**2 + np.abs(ty)**2 + np.abs(tz)**2
        T = np.matmul(np.real(Kz_trn / self.model.ur_trn) / np.real(k_inc[2] / self.model.ur_trn),  t_sqr)
        T = T.reshape((self.x_harms, self.y_harms)).transpose()

        self.model.R_tot = np.sum(R)
        self.model.T_tot = np.sum(T)
        self.model.T = T
        self.model.R = R
        self.model.tx, self.model.ty, self.model.tz = tx, ty, tz
        self.model.rx, self.model.ry, self.model.rz = rx, ry, rz

    def _transmission_S_mat(self, Kx, Ky, Kz_trn, W_0, V_0):
        '''
        Compute the transmission S matrix
        '''
        xy_harms = self.x_harms * self.y_harms
        I = np.eye(xy_harms)
        Q_trn = np.zeros((2*xy_harms, 2*xy_harms)).astype(np.complex)
        Q_trn[:xy_harms, :xy_harms] = Kx * Ky
        Q_trn[:xy_harms, xy_harms:] = self.model.ur_trn * self.model.er_trn * I - Kx**2
        Q_trn[xy_harms:, :xy_harms] = Ky**2 - self.model.ur_trn * self.model.er_trn * I
        Q_trn[xy_harms:, xy_harms:] = 0 - Ky * Kx
        Q_trn = Q_trn / self.model.ur_trn

        W_trn = np.eye(2*xy_harms)
        W_trn = np.mat(W_trn)

        lam_trn = np.zeros_like(W_trn).astype(np.complex)
        lam_trn[:xy_harms, :xy_harms] = 1j * Kz_trn
        lam_trn[xy_harms:, xy_harms:] = 1j * Kz_trn
        V_trn = np.mat(Q_trn) * np.linalg.inv(lam_trn)

        # compute reflectoin side connection scattering matrix
        A_t = np.linalg.inv(W_0) * W_trn + np.linalg.inv(V_0) * V_trn
        B_t = np.linalg.inv(W_0) * W_trn - np.linalg.inv(V_0) * V_trn
        A_t_inv = np.linalg.inv(A_t)

        S_11 = B_t * A_t_inv
        S_12 = 0.5 * (A_t - B_t * A_t_inv * B_t)
        S_21 = 2 * A_t_inv
        S_22 = -A_t_inv * B_t

        S_trn = [S_11, S_12, S_21, S_22]

        return S_trn, W_trn

    def _reflection_S_mat(self, Kx, Ky, Kz_ref, W_0, V_0):
        '''
        Compute the reflectance S matrix
        '''
        xy_harms = self.x_harms * self.y_harms
        I = np.eye(xy_harms)

        Q_ref = np.zeros((2*xy_harms, 2*xy_harms)).astype(np.complex)
        Q_ref[:xy_harms, :xy_harms] = Kx * Ky
        Q_ref[:xy_harms, xy_harms:] = self.model.ur_ref * self.model.er_ref * I - Kx**2
        Q_ref[xy_harms:, :xy_harms] = Ky**2 - self.model.ur_ref * self.model.er_ref * I
        Q_ref[xy_harms:, xy_harms:] = 0 - Ky * Kx
        Q_ref = Q_ref / self.model.ur_ref

        W_ref = np.eye(2*xy_harms)
        W_ref = np.mat(W_ref)

        lam_ref = np.zeros_like(W_ref).astype(np.complex)
        lam_ref[:xy_harms, :xy_harms] = -1j * Kz_ref
        lam_ref[xy_harms:, xy_harms:] = -1j * Kz_ref
        V_ref = np.mat(Q_ref) * np.linalg.inv(lam_ref)

        # compute reflectoin side connection scattering matrix
        A_r = np.linalg.inv(W_0) * W_ref + np.linalg.inv(V_0) * V_ref
        B_r = np.linalg.inv(W_0) * W_ref - np.linalg.inv(V_0) * V_ref
        A_r_inv = np.linalg.inv(A_r)

        S_11 = -A_r_inv * B_r
        S_12 = 2 * A_r_inv
        S_21 = 0.5 * (A_r - B_r * A_r_inv * B_r)
        S_22 = B_r * A_r_inv

        S_ref = [S_11, S_12, S_21, S_22]

        return S_ref, W_ref

    def _device_S_mat(self, S_init, Kx, Ky, W_0, V_0):
        '''
        Calculate the device S matrix by iteratively computing S matrices of all layers
        '''

        xy_harms = self.x_harms * self.y_harms
        I = np.eye(2*xy_harms)
        for i, layer in enumerate(self.model.layers):
            # print(i, layer)

            # build eigen-value problem for the i-th layer
            er_inv = la.inv(np.mat(layer.er_convmat))
            ur_inv = la.inv(np.mat(layer.ur_convmat))

            P_i = np.zeros((2*xy_harms, 2*xy_harms)).astype(np.complex)
            P_i[:xy_harms, :xy_harms] = Kx * er_inv * Ky
            P_i[:xy_harms, xy_harms:] = layer.ur_convmat - Kx * er_inv * Kx
            P_i[xy_harms:, :xy_harms] = Ky * er_inv * Ky - layer.ur_convmat
            P_i[xy_harms:, xy_harms:] = 0 - Ky * er_inv * Kx

            Q_i = np.zeros_like(P_i).astype(np.complex)
            Q_i[:xy_harms, :xy_harms] = Kx * ur_inv * Ky
            Q_i[:xy_harms, xy_harms:] = layer.er_convmat - Kx * ur_inv * Kx
            Q_i[xy_harms:, :xy_harms] = Ky * ur_inv * Ky - layer.er_convmat
            Q_i[xy_harms:, xy_harms:] = 0 - Ky * ur_inv * Kx

            Omega_sq_i = np.matmul(np.mat(P_i), np.mat(Q_i))

            # compute eigen-modes in the i-th layer
            eigen_vals_i, W_i = la.eig(Omega_sq_i)
            eigen_vals_i = np.sqrt(eigen_vals_i)
            eigen_vals_i_inv = np.mat(la.inv(eigen_vals_i.reshape((-1)) * I))

            V_i = W_i * eigen_vals_i_inv
            V_i = Q_i * V_i

            # compute layer scattering matrix for the i-the layer
            A_i = np.linalg.inv(W_i) * W_0 + np.linalg.inv(V_i) * V_0
            B_i = np.linalg.inv(W_i) * W_0 - np.linalg.inv(V_i) * V_0
            X_i = np.exp(-eigen_vals_i * self.source.k_0 * layer.d).reshape(-1) * I
            A_i_inv = np.linalg.inv(A_i)
            B_i_inv = np.linalg.inv(B_i)

            S_11 = np.linalg.inv(A_i - X_i * B_i * A_i_inv * X_i * B_i) * (X_i * B_i * A_i_inv * X_i * A_i - B_i)
            S_12 = np.linalg.inv(A_i - X_i * B_i * A_i_inv * X_i * B_i) * X_i * (A_i - B_i * A_i_inv * B_i)
            S_21 = S_12.copy()
            S_22 = S_11.copy()
            S_i = [S_11, S_12, S_21, S_22]

            S_init = self.RedhefferProd(S_init, S_i)

        return S_init

    def init_device_S_mat(self):
        '''
        Initialize identity matrix for the overall S matrix
        '''
        xy_harms = self.x_harms * self.y_harms
        S11 = np.zeros((2*xy_harms, 2*xy_harms))
        S12 = np.eye(2*xy_harms)
        S21 = np.eye(2*xy_harms)
        S22 = np.zeros((2*xy_harms, 2*xy_harms))

        return [S11, S12, S21, S22]


    def RedhefferProd(self, Sa, Sb):
        '''
        Implementation of Redheffer production, used to compute the S matrix of two
        cascaded devices.
        '''
        I = np.eye(Sa[0].shape[0])
        S11 = Sa[0] + Sa[1] * np.linalg.inv(I - Sb[0] * Sa[3]) * Sb[0] * Sa[2]
        S12 = Sa[1] * np.linalg.inv(I - Sb[0] * Sa[3]) * Sb[1]
        S21 = Sb[2] * np.linalg.inv(I - Sa[3] * Sb[0]) * Sa[2]
        S22 = Sb[3] + Sb[2] * np.linalg.inv(I - Sa[3] * Sb[0]) * Sa[3] * Sb[1]

        return [S11, S12, S21, S22]

    def _eigen_freespace(self, Kx, Ky, Kz):
        '''
        Calculate the eigenvalues of the free space
        '''
        xy_harms = self.x_harms * self.y_harms
        I = np.eye(xy_harms, xy_harms)
        Z = np.zeros_like(I)
        Q = np.zeros((2*xy_harms, 2*xy_harms)).astype(np.complex)

        Q[:xy_harms, :xy_harms] = Kx * Ky
        Q[:xy_harms, xy_harms:] = I - Kx**2
        Q[xy_harms:, :xy_harms] = Ky**2 - I
        Q[xy_harms:, xy_harms:] = Z - Kx * Ky
        W = np.eye(2*xy_harms)

        # lambda
        eigen_vals = np.zeros_like(W).astype(np.complex)
        eigen_vals[:xy_harms, :xy_harms] = 1j * Kz
        eigen_vals[xy_harms:, xy_harms:] = 1j * Kz

        Q = np.mat(Q)
        V = Q * la.inv(np.mat(eigen_vals))
        W_0 = np.mat(W)
        eigen_vals = np.mat(eigen_vals)

        return Q, V, W, eigen_vals

    def _wavevec_expansion(self):
        '''
        Calculate the wave vector expansion
        '''
        I = np.eye(self.x_harms*self.y_harms, self.x_harms*self.y_harms)
        Z = np.zeros_like(I)

        n_ref = np.complex(np.sqrt(self.model.er_ref))
        n_trn = np.complex(np.sqrt(self.model.er_trn))
        k_0 = self.source.k_0
        theta = self.source.theta
        phi = self.source.phi
        k_inc = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) * n_ref

        # orders of diffraction
        p = np.array(range(-int(np.floor(self.x_harms/2)), int(np.floor(self.x_harms/2))+1))
        q = np.array(range(-int(np.floor(self.y_harms/2)), int(np.floor(self.y_harms/2))+1))
        kx = k_inc[0] - 2 * np.pi * p / (k_0 * self.model.Lx)
        ky = k_inc[1] - 2 * np.pi * q / (k_0 * self.model.Ly)
        Kx, Ky = np.meshgrid(kx ,ky)

        Kz_ref = -np.conj(np.sqrt(np.conj(self.model.ur_ref) * np.conj(self.model.er_ref) - Kx**2 - Ky**2))
        Kz_trn = np.conj(np.sqrt(np.conj(self.model.ur_trn) * np.conj(self.model.er_trn) - Kx**2 - Ky**2))
        Kx = Kx.reshape((-1)) * I
        Ky = Ky.reshape((-1)) * I
        Kz_ref = Kz_ref.reshape((-1)) * I
        Kz_trn = Kz_trn.reshape((-1)) * I
        Kz = np.conj(np.sqrt(I - Kx**2 - Ky**2))

        Kx = np.mat(Kx)
        Ky = np.mat(Ky)
        Kz = np.mat(Kz)
        Kz_ref = np.mat(Kz_ref)
        Kz_trn = np.mat(Kz_trn)

        return Kx, Ky, Kz, Kz_ref, Kz_trn

    def _wavevec_expansion_test(self):
        '''
        Test function.
        '''
        I = np.eye(self.x_harms*self.y_harms, self.x_harms*self.y_harms)
        Z = np.zeros_like(I)

        n_ref = np.complex(np.sqrt(self.model.er_ref))
        n_trn = np.complex(np.sqrt(self.model.er_trn))
        k_0 = self.source.k_0
        theta = self.source.theta
        phi = self.source.phi
        k_inc = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) * n_ref

        # orders of diffraction
        p = np.array(range(-int(np.floor(self.x_harms/2)), int(np.floor(self.x_harms/2))+1))
        q = np.array(range(-int(np.floor(self.y_harms/2)), int(np.floor(self.y_harms/2))+1))
        kx = k_inc[0] - 2 * np.pi * p / (k_0 * self.model.Lx)
        ky = k_inc[1] - 2 * np.pi * q / (k_0 * self.model.Ly)
        Kx, Ky = np.meshgrid(kx ,ky)

        Kz_ref = -np.conj(np.sqrt(np.conj(self.model.ur_ref) * np.conj(self.model.er_ref) - Kx**2 - Ky**2))
        Kz_trn = np.conj(np.sqrt(np.conj(self.model.ur_trn) * np.conj(self.model.er_trn) - Kx**2 - Ky**2))
        # print(np.conj(self.model.er_trn))
        # Kx = Kx.reshape((-1)) * I
        # Ky = Ky.reshape((-1)) * I
        # Kz_ref = Kz_ref.reshape((-1)) * I
        # Kz_trn = Kz_trn.reshape((-1)) * I
        # Kz = np.conj(np.sqrt(I - Kx**2 - Ky**2))

        # Kx = np.mat(Kx)
        # Ky = np.mat(Ky)
        # Kz = np.mat(Kz)
        # Kz_ref = np.mat(Kz_ref)
        # Kz_trn = np.mat(Kz_trn)

        # print(Kx, Ky, Kz)
        Kz = 0
        return Kx, Ky, Kz, Kz_ref, Kz_trn, kx, ky, k_inc
