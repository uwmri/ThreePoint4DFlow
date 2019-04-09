import numpy as np

class MRI_4DFlow:

    EncodingMatrix =np.pi/2.0*np.array([[-1.0, -1.0, -1.0],
                               [ 1.0, -1.0, -1.0],
                               [-1.0,  1.0, -1.0],
                               [-1.0, -1.0,  1.0]],dtype=np.float32)
    DecodingMatrix = np.linalg.pinv(EncodingMatrix)
    Venc = 1.0  #m/s
    NoiseLevel = 0.0 #relative to max signal of 1
    spatial_resolution = 0.5 # percent of kmax
    time_resolution = 0.5 # percent of nominal
    background_magnitude = 0.5 #value of background

    # Matrices
    signal = None
    velocity_estimate = None


    def set_encoding_matrix(self, encode_type='4pt-referenced'):
        encode_dictionary = {
            '4pt-referenced' : np.pi/2.0*np.array([[-1.0, -1.0, -1.0],
                               [ 1.0, -1.0, -1.0],
                               [-1.0,  1.0, -1.0],
                               [-1.0, -1.0,  1.0]],dtype=np.float32),
            '3pt': np.pi / 2.0 * np.array([[-1.0, -1.0, -1.0],
                                                      [1.0, -1.0, -1.0],
                                                      [-1.0, 1.0, -1.0]], dtype=np.float32),
            '4pt-balanced': np.pi / 2.0/ np.sqrt(2.0) * np.array([[-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32),
            '5pt': np.pi / np.sqrt(3.0) * np.array([ [0.0, 0.0, 0.0],
                                                      [-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32)
        }
        self.EncodingMatrix = encode_dictionary[encode_type]
        self.DecodingMatrix = np.linalg.pinv(self.EncodingMatrix)

    """
    :param velocity: a Nt x Nz x Ny x Nx x 3 description of the velocity field
    :param pd: a Nt x Nz x Ny x Nx mask of the vessel locations
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def get_complex_signal(self,velocity,pd):

        # Get last dimension to (3 x 1)
        velocity = np.expand_dims( velocity,-1)

        # Multiple to get phase
        print(self.EncodingMatrix.shape)
        print(velocity.shape)

        # Get the Phase
        phase = np.matmul( self.EncodingMatrix/self.Venc, velocity)

        # Create Magnitude image (M*exp(i*phase))
        mag = np.copy(pd)
        mag += self.background_magnitude
        mag = np.expand_dims(mag, -1)
        mag = np.expand_dims(mag, -1)
        self.signal = mag*np.exp(1j * phase )

        print(self.signal.shape)


    def solve_for_velocity(self):

        # Multiply by reference
        ref = self.signal[...,0,0]
        ref = np.expand_dims(ref, -1)
        ref = np.expand_dims(ref, -1)
        signal2 = self.signal * np.conj(ref)

        # Get subtracted decoding matrix
        diffMatrix = self.EncodingMatrix
        diffMatrix -= diffMatrix[0,:]
        self.DecodingMatrix = np.linalg.pinv(diffMatrix)
        #print(self.DecodingMatrix)

        # Take angle
        phase = np.angle(signal2)

        #print(phase.shape)

        #Solve for velocity
        self.velocity_estimate = np.matmul(self.DecodingMatrix*self.Venc,phase)

